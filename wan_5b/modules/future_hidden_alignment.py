import torch
from torch import nn
from torch.nn import functional as F


class _FutureHiddenAttention(nn.Module):
    """Full self-attention over one Current chunk, without added positions."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dim = hidden_states.shape
        qkv = self.qkv(hidden_states).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
        )
        return self.out(attended.transpose(1, 2).reshape(batch_size, sequence_length, dim))


class _FutureHiddenDiTBlock(nn.Module):
    """Position-free pre-norm Transformer block used only during training."""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float) -> None:
        super().__init__()
        ffn_dim = int(dim * ffn_ratio)
        if ffn_dim <= 0:
            raise ValueError("ffn_ratio must produce a positive FFN dimension.")
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = _FutureHiddenAttention(dim, num_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        return hidden_states + self.ffn(self.ffn_norm(hidden_states))


class FutureHiddenAlignmentHead(nn.Module):
    """Predict clean next-chunk features from noisy Current hidden states."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        projector_type: str = "mlp",
        num_layers: int = 3,
        num_heads: int = 24,
        ffn_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.projector_type = projector_type.lower()
        if self.projector_type == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim),
            )
            self.blocks = nn.ModuleList()
            self.output = nn.Identity()
        elif self.projector_type == "dit":
            if num_layers <= 0:
                raise ValueError("Future-hidden DiT must contain at least one block.")
            self.net = nn.Identity()
            self.blocks = nn.ModuleList(
                [
                    _FutureHiddenDiTBlock(dim, num_heads, ffn_ratio)
                    for _ in range(num_layers)
                ]
            )
            self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        else:
            raise ValueError(
                f"Unsupported future-hidden projector_type={projector_type!r}; "
                "expected 'mlp' or 'dit'."
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError("hidden_states must have shape [B, P, N, D].")
        if self.projector_type == "mlp":
            return self.net(hidden_states)

        batch_size, num_pairs, num_tokens, dim = hidden_states.shape
        prediction = hidden_states.flatten(0, 1)
        for block in self.blocks:
            prediction = block(prediction)
        prediction = self.output(prediction)
        return prediction.unflatten(0, (batch_size, num_pairs)).reshape(
            batch_size, num_pairs, num_tokens, dim
        )


def extract_next_chunk_hidden(
    hidden_states: torch.Tensor,
    *,
    num_clean_frames: int,
    num_current_frames: int,
    num_pairs: int,
    block_size: int,
    frame_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair each noisy Current chunk with the following clean-history chunk."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, N, D].")
    if min(num_clean_frames, num_current_frames, block_size, frame_seqlen) <= 0:
        raise ValueError("Future-hidden alignment dimensions must be positive.")
    if num_pairs < 0:
        raise ValueError("Future-hidden alignment num_pairs must be non-negative.")

    clean_tokens = num_clean_frames * frame_seqlen
    current_tokens = num_current_frames * frame_seqlen
    required_frames = (num_pairs + 1) * block_size
    if required_frames > num_clean_frames or num_pairs * block_size > num_current_frames:
        raise ValueError(
            "Not enough clean/current frames for next-chunk alignment: "
            f"clean={num_clean_frames}, current={num_current_frames}, "
            f"pairs={num_pairs}, block_size={block_size}."
        )
    if hidden_states.shape[1] < clean_tokens + current_tokens:
        raise ValueError("hidden_states is shorter than the clean and Current streams.")

    tokens_per_chunk = block_size * frame_seqlen
    current_start = clean_tokens
    student = hidden_states[
        :, current_start:current_start + num_pairs * tokens_per_chunk
    ]
    teacher = hidden_states[
        :, tokens_per_chunk:(num_pairs + 1) * tokens_per_chunk
    ]
    return (
        student.unflatten(1, (num_pairs, tokens_per_chunk)),
        teacher.unflatten(1, (num_pairs, tokens_per_chunk)),
    )


def masked_future_hidden_cosine_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid_tokens: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean token-wise cosine distance over valid next-chunk tokens."""
    if student.shape != teacher.shape or student.ndim != 4:
        raise ValueError(
            "student and teacher must have identical [B, P, N, D] shapes, got "
            f"{tuple(student.shape)} and {tuple(teacher.shape)}."
        )
    if valid_tokens.shape != student.shape[:-1]:
        raise ValueError(
            f"valid_tokens must have shape {tuple(student.shape[:-1])}, "
            f"got {tuple(valid_tokens.shape)}."
        )

    cosine_distance = 1.0 - F.cosine_similarity(
        student.float(), teacher.detach().float(), dim=-1
    )
    weights = valid_tokens.to(device=cosine_distance.device, dtype=cosine_distance.dtype)
    if token_weights is not None:
        if token_weights.shape != weights.shape:
            raise ValueError(
                f"token_weights must have shape {tuple(weights.shape)}, "
                f"got {tuple(token_weights.shape)}."
            )
        token_weights = token_weights.to(device=weights.device, dtype=weights.dtype)
        if not torch.isfinite(token_weights).all() or (token_weights < 0).any():
            raise ValueError("token_weights must be finite and non-negative.")
        weights = weights * token_weights
    return (cosine_distance * weights).sum() / weights.sum().clamp_min(1e-12)


def future_hidden_pixel_weights(
    raw_video: torch.Tensor,
    *,
    num_latent_frames: int,
    num_pairs: int,
    block_size: int,
    spatial_size: tuple[int, int],
    mode: str,
    background_weight: float,
    changed_weight: float = 0.6,
    pixel_threshold: float = 0.02,
) -> torch.Tensor:
    """Map Current-to-Next RGB changes to aligned future-hidden token weights."""
    if raw_video.ndim != 5:
        raise ValueError("raw_video must have shape [B, C, T, H, W].")
    if min(num_latent_frames, block_size, *spatial_size) <= 0 or num_pairs < 0:
        raise ValueError("Pixel-weight dimensions must be positive.")
    if num_latent_frames <= 1 or (raw_video.shape[2] - 1) % (num_latent_frames - 1):
        raise ValueError(
            "Cannot infer an integer latent-to-pixel temporal ratio from "
            f"{num_latent_frames} latent and {raw_video.shape[2]} pixel frames."
        )
    if (num_pairs + 1) * block_size > num_latent_frames:
        raise ValueError("Not enough latent frames for every Current-to-Next pair.")
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError("background_weight must be in [0, 1].")

    mode = mode.lower()
    if mode not in ("continuous", "binary"):
        raise ValueError(f"Unsupported pixel weighting mode={mode!r}.")
    if mode == "binary":
        if not background_weight < changed_weight <= 1.0:
            raise ValueError("changed_weight must be in (background_weight, 1].")
        if not 0.0 <= pixel_threshold <= 1.0:
            raise ValueError("pixel_threshold must be in [0, 1].")
    if num_pairs == 0:
        return raw_video.new_empty(
            raw_video.shape[0],
            0,
            block_size * spatial_size[0] * spatial_size[1],
            dtype=torch.float32,
        )

    temporal_ratio = (raw_video.shape[2] - 1) // (num_latent_frames - 1)
    current_indices = torch.arange(
        num_pairs * block_size, device=raw_video.device, dtype=torch.long
    )
    next_indices = current_indices + block_size
    current_frames = raw_video.index_select(2, current_indices * temporal_ratio)
    next_frames = raw_video.index_select(2, next_indices * temporal_ratio)

    # Dataset pixels are normalized to [-1, 1]; divide their absolute change by 2
    # so the threshold is expressed in the familiar [0, 1] RGB scale.
    change = (next_frames.float() - current_frames.float()).abs().mean(dim=1) * 0.5
    change = change.flatten(0, 1).unsqueeze(1)
    pooled = F.adaptive_avg_pool2d(change, spatial_size).squeeze(1)
    pooled = pooled.unflatten(0, (raw_video.shape[0], num_pairs, block_size))

    if mode == "binary":
        weights = torch.where(
            pooled > pixel_threshold,
            torch.full_like(pooled, changed_weight),
            torch.full_like(pooled, background_weight),
        )
    else:
        scale = pooled.flatten(-2).amax(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = pooled / scale.unsqueeze(-1)
        weights = background_weight + (1.0 - background_weight) * normalized
    return weights.flatten(2)
