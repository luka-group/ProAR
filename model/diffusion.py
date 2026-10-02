# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import random
import torch

from model.base import BaseModel
from pipeline import CausalDiffusionInferencePipeline
from utils.i2v_conditioning import _overwrite_i2v_context, _zero_i2v_context_timestep
from utils.lookahead import (
    goal_is_strictly_future,
    lookahead_is_active_for_chunk,
    resolve_goal_frame_index,
    strictly_future_goal_pair_count,
    validate_lookahead_start_chunk,
)
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan_5b.modules.future_hidden_alignment import (
    future_hidden_pixel_weights,
    masked_future_hidden_cosine_loss,
)


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame and not getattr(args, "i2v", False):
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        self.future_lookahead = bool(getattr(args, "future_lookahead", False))
        configured_lookahead_layout = getattr(args, "lookahead_layout", "endpoint_replicas")
        self.lookahead_next_chunk_last_frame = False
        self.lookahead_layout = configured_lookahead_layout
        self.lookahead_start_chunk = validate_lookahead_start_chunk(
            getattr(args, "lookahead_start_chunk", 0)
        )
        if self.lookahead_start_chunk > 0 and self.lookahead_layout != "goal_frame":
            raise ValueError("Delayed lookahead currently requires lookahead_layout=goal_frame.")
        self.lookahead_current_attends_endpoint = bool(
            getattr(
                args,
                "lookahead_current_attends_endpoint",
                self.lookahead_layout == "endpoint_replicas",
            )
        )
        self.skip_first_lookahead_endpoint = bool(
            getattr(args, "skip_first_lookahead_endpoint", False)
        )
        self.lookahead_first_current_attends_endpoint = bool(
            getattr(
                args,
                "lookahead_first_current_attends_endpoint",
                not self.skip_first_lookahead_endpoint,
            )
        )
        self.endpoint_loss_weight = float(getattr(args, "endpoint_loss_weight", 0.5))
        self.lookahead_goal_attends_current = bool(
            getattr(args, "lookahead_goal_attends_current", True)
        )
        self.lookahead_goal_attends_i2v_context = bool(
            getattr(
                args,
                "lookahead_goal_attends_i2v_context",
                bool(getattr(args, "i2v", False))
                and self.lookahead_layout == "goal_frame"
                and not self.lookahead_goal_attends_current,
            )
        )
        configured_goal_frame_index = getattr(args, "lookahead_goal_frame_index", None)
        self.lookahead_goal_frame_index = (
            int(configured_goal_frame_index)
            if configured_goal_frame_index is not None else None
        )
        self.lookahead_stop_when_goal_not_future = bool(
            getattr(args, "lookahead_stop_when_goal_not_future", False)
        )
        self.endpoint_loss_sigma_low = getattr(args, "endpoint_loss_sigma_low", None)
        self.endpoint_loss_sigma_high = getattr(args, "endpoint_loss_sigma_high", None)
        if self.endpoint_loss_sigma_low is not None:
            self.endpoint_loss_sigma_low = float(self.endpoint_loss_sigma_low)
        if self.endpoint_loss_sigma_high is not None:
            self.endpoint_loss_sigma_high = float(self.endpoint_loss_sigma_high)
        self.share_endpoint_noise_across_replicas = bool(
            getattr(args, "share_endpoint_noise_across_replicas", True)
        )
        future_hidden_cfg = getattr(args, "future_hidden_alignment", None)
        self.future_hidden_alignment_enabled = bool(
            future_hidden_cfg is not None
            and getattr(future_hidden_cfg, "enabled", False)
        )
        self.future_hidden_alignment_weight = float(
            getattr(future_hidden_cfg, "loss_weight", 0.01)
            if future_hidden_cfg is not None else 0.01
        )
        self.future_hidden_pixel_weighting = str(
            getattr(future_hidden_cfg, "pixel_weighting", "none")
            if future_hidden_cfg is not None else "none"
        ).lower()
        self.future_hidden_pixel_spatial_size = tuple(
            int(value) for value in (
                getattr(future_hidden_cfg, "pixel_spatial_size", (16, 16))
                if future_hidden_cfg is not None else (16, 16)
            )
        )
        self.future_hidden_pixel_background_weight = float(
            getattr(future_hidden_cfg, "pixel_background_weight", 0.25)
            if future_hidden_cfg is not None else 0.25
        )
        self.future_hidden_pixel_changed_weight = float(
            getattr(future_hidden_cfg, "pixel_changed_weight", 0.6)
            if future_hidden_cfg is not None else 0.6
        )
        self.future_hidden_pixel_threshold = float(
            getattr(future_hidden_cfg, "pixel_threshold", 0.02)
            if future_hidden_cfg is not None else 0.02
        )
        if self.future_hidden_alignment_enabled:
            if not self.teacher_forcing:
                raise ValueError("future_hidden_alignment requires teacher_forcing=true.")
            if self.future_hidden_pixel_weighting not in ("none", "continuous", "binary"):
                raise ValueError(
                    "future_hidden_alignment.pixel_weighting must be one of "
                    "'none', 'continuous', or 'binary'."
                )
            self.generator.model.configure_future_hidden_alignment(
                student_layer=int(future_hidden_cfg.student_layer),
                teacher_layer=int(future_hidden_cfg.teacher_layer),
                projector_hidden_dim=int(
                    getattr(future_hidden_cfg, "projector_hidden_dim", 4096)
                ),
                projector_type=str(getattr(future_hidden_cfg, "projector_type", "mlp")),
                num_layers=int(getattr(future_hidden_cfg, "num_layers", 3)),
                num_heads=int(getattr(future_hidden_cfg, "num_heads", 24)),
                ffn_ratio=float(getattr(future_hidden_cfg, "ffn_ratio", 2.0)),
            )
        if self.future_lookahead:
            if self.lookahead_layout not in ("endpoint_replicas", "goal_frame"):
                raise ValueError(f"Unsupported lookahead_layout: {self.lookahead_layout}")
            if not self.teacher_forcing:
                raise ValueError("future_lookahead requires teacher_forcing=true.")
            if int(getattr(args, "sequence_parallel_size", 1) or 1) != 1:
                raise ValueError("future_lookahead currently requires sequence_parallel_size=1.")
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

        self.args = args
        self.device = device
        self.inference_pipeline = None

        # Error recycling (SVI-style error buffer)
        # When ``enable_position_bucketing`` is true, each rank holds a 2D
        # buffer ``(local_block_position × timestep)``. The pos dimension only
        # covers the LOCAL slice of the sequence this rank is responsible for
        # (no cross-SP-rank pos sharing — those positions are simply not
        # reachable by this rank during forward), so memory cost scales as
        # ``num_blocks_global / sp_size`` instead of ``num_blocks_global``.
        # ``global_block_offset`` is recorded for logging only.
        # During the first ``buffer_warmup_iter`` global steps, errors are
        # all-gathered across the DP group (ranks with the same SP rank but
        # different DP replicas), so each rank's local pos buckets fill up
        # ``dp_size`` × faster without any wasted bandwidth.
        self.error_buffer = None
        self.noise_error_buffer = None
        self.er_num_blocks = 0       # local; >0 means 2D position-bucketed
        self.er_block_offset = 0     # global block offset for THIS rank
        er_cfg = getattr(args, "error_recycling", None)
        if er_cfg is not None and getattr(er_cfg, "enabled", False):
            from utils.error_buffer import build_error_buffer
            cfg_dict = er_cfg if isinstance(er_cfg, dict) else dict(er_cfg)
            cfg_dict.setdefault("num_train_timesteps", self.num_train_timestep)
            sp_size = int(getattr(args, "sequence_parallel_size", 1) or 1)
            if cfg_dict.get("enable_position_bucketing", False):
                shape = list(getattr(args, "image_or_video_shape", [1, 0]))
                total_frames = int(shape[1]) if len(shape) > 1 else 0
                assert total_frames > 0 and self.num_frame_per_block > 0, (
                    "enable_position_bucketing=true requires "
                    "image_or_video_shape[1] and num_frame_per_block to be set."
                )
                num_blocks_global = total_frames // self.num_frame_per_block
                assert num_blocks_global % sp_size == 0, (
                    f"num_blocks_global ({num_blocks_global}) must be divisible "
                    f"by sequence_parallel_size ({sp_size})."
                )
                self.er_num_blocks = num_blocks_global // sp_size  # local
                # Determine this rank's SP index → global block offset.
                if sp_size > 1:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        sp_rank = dist.get_rank() % sp_size
                    else:
                        sp_rank = 0
                else:
                    sp_rank = 0
                self.er_block_offset = sp_rank * self.er_num_blocks

            # Shard timestep buckets across SP ranks: each SP rank only
            # stores t_bucket % sp_size == sp_rank, cutting per-rank CPU
            # memory by ~sp_size.  This uses the same SP dimension that
            # already splits positions in 2D mode, so both 1D and 2D
            # follow one save/load pattern (per sp_rank).
            import torch.distributed as dist
            if sp_size > 1 and dist.is_initialized():
                er_shard_rank = dist.get_rank() % sp_size
                er_shard_size = sp_size
            else:
                er_shard_rank = 0
                er_shard_size = 1

            self.error_buffer = build_error_buffer(
                cfg_dict, num_blocks=self.er_num_blocks,
                global_block_offset=self.er_block_offset,
                shard_rank=er_shard_rank, shard_size=er_shard_size,
            )
            self.noise_error_buffer = build_error_buffer(
                cfg_dict, num_blocks=self.er_num_blocks,
                global_block_offset=self.er_block_offset,
                shard_rank=er_shard_rank, shard_size=er_shard_size,
            )
            self.er_context_inject_prob = float(cfg_dict.get("context_inject_prob", 0.9))
            self.er_latent_inject_prob = float(cfg_dict.get("latent_inject_prob", 0.0))
            self.er_noise_inject_prob = float(cfg_dict.get("noise_inject_prob", 0.0))
            self.er_clean_prob = float(cfg_dict.get("clean_prob", 0.0))
            self.er_clean_buffer_update_prob = float(cfg_dict.get("clean_buffer_update_prob", 0.1))
            self.er_start_step = int(cfg_dict.get("start_step", 0))
            self.er_buffer_warmup_iter = int(cfg_dict.get("buffer_warmup_iter", 50))
            self.er_skip_block_0 = bool(cfg_dict.get("skip_block_0", False))

    def _initialize_models(self, args, device):
        model_name = getattr(args.model_kwargs, "model_name", "Wan2.2-TI2V-5B")
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported in this release, got {model_name}")
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _future_hidden_token_weights(
        self,
        raw_video: torch.Tensor,
        *,
        num_frames: int,
        num_pairs: int,
        block_size: int,
        frame_seqlen: int,
    ) -> torch.Tensor | None:
        if self.future_hidden_pixel_weighting == "none":
            return None
        if raw_video is None:
            raise ValueError("Pixel-weighted future-hidden alignment requires raw_video.")
        if len(self.future_hidden_pixel_spatial_size) != 2:
            raise ValueError("pixel_spatial_size must contain exactly [height, width].")
        num_spatial_tokens = (
            self.future_hidden_pixel_spatial_size[0]
            * self.future_hidden_pixel_spatial_size[1]
        )
        if num_spatial_tokens != frame_seqlen:
            raise ValueError(
                f"pixel_spatial_size={self.future_hidden_pixel_spatial_size} contains "
                f"{num_spatial_tokens} "
                f"tokens, but the model has frame_seqlen={frame_seqlen}."
            )
        return future_hidden_pixel_weights(
            raw_video,
            num_latent_frames=num_frames,
            num_pairs=num_pairs,
            block_size=block_size,
            spatial_size=self.future_hidden_pixel_spatial_size,
            mode=self.future_hidden_pixel_weighting,
            background_weight=self.future_hidden_pixel_background_weight,
            changed_weight=self.future_hidden_pixel_changed_weight,
            pixel_threshold=self.future_hidden_pixel_threshold,
        )

    def future_lookahead_generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        loss_mask_global_valid_count: torch.Tensor = None,
        global_step: int = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frame = image_or_video_shape[:2]
        block_size = self.num_frame_per_block
        if num_frame % block_size != 0:
            raise ValueError(
                f"future_lookahead requires num_frame ({num_frame}) to be divisible by "
                f"num_frame_per_block ({block_size})."
            )

        num_blocks = num_frame // block_size
        endpoint_block = num_blocks - 1
        if endpoint_block <= 0:
            raise ValueError("future_lookahead requires at least two chunks.")
        endpoint_start = endpoint_block * block_size
        endpoint_end = endpoint_start + block_size
        endpoint_pair_start_block = 0
        num_pairs = endpoint_block - endpoint_pair_start_block
        if num_pairs <= 0:
            raise ValueError("future_lookahead has no endpoint pairs.")
        endpoint_replica_frames = num_pairs * block_size

        index_current = self._get_timestep( #current chunks 正常采样 timestep
            0,
            self.scheduler.num_train_timesteps,
            batch_size,
            num_frame,
            block_size,
            uniform_timestep=False
        )
        timestep_current = self.scheduler.timesteps[index_current].to(
            dtype=self.dtype, device=self.device
        )

        context_latent = (
            initial_latent
            if getattr(self.args, "i2v", False) and initial_latent is not None
            else None
        )
        context_frames = int(context_latent.shape[1]) if context_latent is not None else 0
        if context_frames > 0:
            if context_frames >= num_frame:
                raise ValueError(
                    f"initial_latent has {context_frames} frames but training clip has {num_frame}."
                )
            timestep_current[:, :context_frames] = 0

        noise_current = torch.randn_like(clean_latent)
        noisy_current = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise_current.flatten(0, 1),
            timestep_current.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        target_current = self.scheduler.training_target(clean_latent, noise_current, timestep_current)

        if context_frames > 0:
            noisy_current[:, :context_frames] = context_latent.to(
                device=noisy_current.device,
                dtype=noisy_current.dtype,
            )
            target_current[:, :context_frames] = 0.

        endpoint_clean = clean_latent[:, endpoint_start:endpoint_end]
        endpoint_index_blocks = index_current[:, endpoint_pair_start_block * block_size:endpoint_start:block_size] # endpoint replica 的 timestep 取对应 current chunk 的 timestep
        endpoint_index = endpoint_index_blocks.repeat_interleave(block_size, dim=1)
        endpoint_timestep = self.scheduler.timesteps[endpoint_index].to(
            dtype=self.dtype, device=self.device
        )
        endpoint_clean_replicas = endpoint_clean.repeat(1, num_pairs, 1, 1, 1)
        if self.share_endpoint_noise_across_replicas: #current和last的noise epsilon 不同；noise epsilon采样noise相同
            endpoint_noise = torch.randn_like(endpoint_clean).repeat(1, num_pairs, 1, 1, 1)
        else:
            endpoint_noise = torch.randn_like(endpoint_clean_replicas)
        noisy_endpoint = self.scheduler.add_noise(
            endpoint_clean_replicas.flatten(0, 1),
            endpoint_noise.flatten(0, 1),
            endpoint_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, endpoint_replica_frames))
        target_endpoint = self.scheduler.training_target(
            endpoint_clean_replicas, endpoint_noise, endpoint_timestep
        )
# e.g.
# noisy_current:  原视频 24 latent frames
# noisy_endpoint: last chunk replicas, 5 * 4 = 20 latent frames
# total noisy input: 44 latent frames
        noisy_model_input = torch.cat([noisy_current, noisy_endpoint], dim=1)
        timestep_model_input = torch.cat([timestep_current, endpoint_timestep], dim=1)
        target_model = torch.cat([target_current, target_endpoint], dim=1)

        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                batch_size,
                num_frame,
                block_size,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(
                dtype=self.dtype, device=self.device
            )
            clean_aug_noise = torch.randn_like(clean_latent)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                clean_aug_noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = torch.zeros(
                (batch_size, num_frame),
                device=self.device,
                dtype=timestep_current.dtype,
            )

        if context_frames > 0:
            clean_latent_aug = _overwrite_i2v_context(
                clean_latent_aug, context_latent, context_frames
            )
            timestep_clean_aug = _zero_i2v_context_timestep(
                timestep_clean_aug, context_frames
            )

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_model_input,
            conditional_dict=conditional_dict,
            timestep=timestep_model_input,
            clean_x=clean_latent_aug,
            aug_t=timestep_clean_aug,
            lookahead_layout=self.lookahead_layout,
            lookahead_num_frames=num_frame,
            lookahead_endpoint_start_frame=endpoint_start,
            lookahead_endpoint_replica_frames=endpoint_replica_frames,
            lookahead_endpoint_pair_start_block=endpoint_pair_start_block,
            lookahead_current_attends_endpoint=self.lookahead_current_attends_endpoint,
            lookahead_first_current_attends_endpoint=self.lookahead_first_current_attends_endpoint,
        )

        per_frame_loss = torch.nn.functional.mse_loss(
            flow_pred.float(), target_model.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        per_frame_loss = per_frame_loss * self.scheduler.training_weight(
            timestep_model_input
        ).unflatten(0, (batch_size, num_frame + endpoint_replica_frames))

        if loss_mask is None:
            current_mask = torch.ones(
                (batch_size, num_frame),
                device=per_frame_loss.device,
                dtype=per_frame_loss.dtype,
            )
        else:
            current_mask = loss_mask.to(device=per_frame_loss.device, dtype=per_frame_loss.dtype).clone()
        if context_frames > 0:
            current_mask[:, :context_frames] = 0

        endpoint_base_mask = current_mask.new_ones((batch_size, block_size))
        if loss_mask is not None:
            endpoint_base_mask = loss_mask[:, endpoint_start:endpoint_end].to(
                device=per_frame_loss.device,
                dtype=per_frame_loss.dtype,
            )
        endpoint_mask = endpoint_base_mask.repeat(1, num_pairs)
        if self.endpoint_loss_sigma_low is not None:
            sigma_low = self.endpoint_loss_sigma_low
            sigma_high = self.endpoint_loss_sigma_high
            if sigma_high is None:
                sigma_high = sigma_low
            endpoint_sigma = self.scheduler.sigmas[endpoint_index].to(
                device=per_frame_loss.device,
                dtype=per_frame_loss.dtype,
            )
            if sigma_high <= sigma_low:
                endpoint_gate = (endpoint_sigma >= sigma_low).to(endpoint_mask.dtype)
            else:
                endpoint_gate = ((endpoint_sigma - sigma_low) / (sigma_high - sigma_low)).clamp(0.0, 1.0)
            endpoint_mask = endpoint_mask * endpoint_gate
#################################
        current_loss = per_frame_loss[:, :num_frame]
        endpoint_loss = per_frame_loss[:, num_frame:]
        current_valid = current_mask.sum().clamp(min=1.0)
        endpoint_valid = endpoint_mask.sum().clamp(min=1.0)
        current_loss_mean = (current_loss * current_mask).sum() / current_valid
        endpoint_loss_mean = (endpoint_loss * endpoint_mask).sum() / endpoint_valid
        loss = current_loss_mean + self.endpoint_loss_weight * endpoint_loss_mean
#################################
        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred[:, :num_frame].detach(),
            "lookahead_current_loss": current_loss_mean.detach(),
            "lookahead_endpoint_loss": endpoint_loss_mean.detach(),
            "lookahead_endpoint_loss_weight": torch.tensor(
                self.endpoint_loss_weight,
                device=loss.device,
                dtype=loss.dtype,
            ),
            "lookahead_endpoint_active_ratio": (
                (endpoint_mask > 0).float().mean().detach()
            ),
        }
        return loss, log_dict

    def goal_frame_lookahead_generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        loss_mask_global_valid_count: torch.Tensor = None,
        global_step: int = None,
        raw_video: torch.Tensor = None,
        goal_frame_index: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frame = image_or_video_shape[:2]
        block_size = self.num_frame_per_block
        if batch_size != 1:
            raise ValueError("goal_frame lookahead currently expects batch_size=1 per rank.")
        if num_frame % block_size != 0:
            raise ValueError(
                f"goal_frame requires num_frame ({num_frame}) to be divisible by "
                f"num_frame_per_block ({block_size})."
            )

        if loss_mask is None:
            valid_t = num_frame
            valid_mask = torch.ones(
                (batch_size, num_frame), device=self.device, dtype=torch.float32
            )
        else:
            valid_mask = loss_mask.to(device=self.device, dtype=torch.float32)
            valid_indices = torch.nonzero(valid_mask[0] > 0, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                raise ValueError("goal_frame lookahead has no supervised video frames.")
            # I2V masks frame 0 because it is the provided context. The last
            # nonzero position, not mask.sum(), therefore defines the true end.
            valid_t = int(valid_indices[-1].item()) + 1
        if valid_t <= block_size:
            raise ValueError("goal_frame lookahead requires at least two effective chunks.")

        max_num_pairs = (valid_t - 1) // block_size
        if max_num_pairs <= 0:
            raise ValueError("goal_frame lookahead has no current-goal pairs.")

        if self.lookahead_next_chunk_last_frame:
            num_pairs = max_num_pairs
            pair_ids = torch.arange(num_pairs, device=self.device, dtype=torch.long)
            goal_frame_indices = [
                min((pair_id + 2) * block_size - 1, valid_t - 1)
                for pair_id in range(num_pairs)
            ]
            goal_indices = torch.tensor(
                goal_frame_indices, device=self.device, dtype=torch.long
            )
        else:
            requested_goal_frame_index = (
                int(goal_frame_index[0].item())
                if goal_frame_index is not None else None
            )
            goal_idx = resolve_goal_frame_index(
                configured_goal_frame_index=self.lookahead_goal_frame_index,
                requested_goal_frame_index=requested_goal_frame_index,
                fallback_goal_frame_index=valid_t - 1,
                num_frames=valid_t,
            )
            num_pairs = max_num_pairs
            if self.lookahead_stop_when_goal_not_future:
                num_pairs = strictly_future_goal_pair_count(valid_t, goal_idx, block_size)
            if num_pairs <= 0:
                raise ValueError("goal_frame lookahead has no strictly-future goal pairs.")
            goal_frame_indices = None
            goal_indices = torch.full((num_pairs,), goal_idx, device=self.device, dtype=torch.long)

        index_current = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            batch_size,
            num_frame,
            block_size,
            uniform_timestep=False
        )
        timestep_current = self.scheduler.timesteps[index_current].to(
            dtype=self.dtype, device=self.device
        )

        context_latent = (
            initial_latent
            if getattr(self.args, "i2v", False) and initial_latent is not None
            else None
        )
        context_frames = int(context_latent.shape[1]) if context_latent is not None else 0
        if context_frames > 0:
            if context_frames >= num_frame:
                raise ValueError(
                    f"initial_latent has {context_frames} frames but training clip has {num_frame}."
                )
            timestep_current[:, :context_frames] = 0

        noise_current = torch.randn_like(clean_latent)
        noisy_current = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise_current.flatten(0, 1),
            timestep_current.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        target_current = self.scheduler.training_target(clean_latent, noise_current, timestep_current)

        if context_frames > 0:
            noisy_current[:, :context_frames] = context_latent.to(
                device=noisy_current.device,
                dtype=noisy_current.dtype,
            )
            target_current[:, :context_frames] = 0.

        pair_index_blocks = index_current[:, :num_pairs * block_size:block_size]
        pair_timestep = self.scheduler.timesteps[pair_index_blocks].to(
            dtype=self.dtype, device=self.device
        )

        goal_clean = clean_latent.index_select(1, goal_indices)
        if self.share_endpoint_noise_across_replicas and not self.lookahead_next_chunk_last_frame:
            goal_noise = torch.randn_like(clean_latent[:, goal_idx:goal_idx + 1]).repeat(
                1, num_pairs, 1, 1, 1
            )
        else:
            goal_noise = torch.randn_like(goal_clean)
        noisy_goal = self.scheduler.add_noise(
            goal_clean.flatten(0, 1),
            goal_noise.flatten(0, 1),
            pair_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_pairs))
        target_goal = self.scheduler.training_target(goal_clean, goal_noise, pair_timestep)

        noisy_model_input = torch.cat([noisy_current, noisy_goal], dim=1)
        timestep_model_input = torch.cat([timestep_current, pair_timestep], dim=1)
        target_model = torch.cat([target_current, target_goal], dim=1)

        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                batch_size,
                num_frame,
                block_size,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(
                dtype=self.dtype, device=self.device
            )
            clean_aug_noise = torch.randn_like(clean_latent)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                clean_aug_noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = torch.zeros(
                (batch_size, num_frame),
                device=self.device,
                dtype=timestep_current.dtype,
            )

        if context_frames > 0:
            clean_latent_aug = _overwrite_i2v_context(
                clean_latent_aug, context_latent, context_frames
            )
            timestep_clean_aug = _zero_i2v_context_timestep(
                timestep_clean_aug, context_frames
            )

        generator_output = self.generator(
            noisy_image_or_video=noisy_model_input,
            conditional_dict=conditional_dict,
            timestep=timestep_model_input,
            clean_x=clean_latent_aug,
            aug_t=timestep_clean_aug,
            lookahead_layout=self.lookahead_layout,
            lookahead_num_frames=num_frame,
            lookahead_endpoint_start_frame=(
                goal_frame_indices[-1]
                if self.lookahead_next_chunk_last_frame
                else goal_idx
            ),
            lookahead_endpoint_frame_indices=goal_frame_indices,
            lookahead_endpoint_replica_frames=num_pairs,
            lookahead_endpoint_pair_start_block=0,
            lookahead_current_attends_endpoint=self.lookahead_current_attends_endpoint,
            lookahead_first_current_attends_endpoint=(
                self.lookahead_first_current_attends_endpoint
                and lookahead_is_active_for_chunk(0, self.lookahead_start_chunk)
            ),
            lookahead_goal_attends_current=self.lookahead_goal_attends_current,
            lookahead_goal_attends_i2v_context=self.lookahead_goal_attends_i2v_context,
        )
        future_hidden_prediction = None
        future_hidden_target = None
        if self.future_hidden_alignment_enabled:
            if len(generator_output) != 3:
                raise RuntimeError("Auxiliary supervision expected generator hidden-state predictions.")
            flow_pred, x0_pred, auxiliary_prediction = generator_output
            if isinstance(auxiliary_prediction, dict):
                future_hidden_prediction = auxiliary_prediction.get("future_hidden_prediction")
                future_hidden_target = auxiliary_prediction.get("future_hidden_target")
            else:
                raise RuntimeError("Future-hidden alignment expected an auxiliary prediction dictionary.")
        else:
            flow_pred, x0_pred = generator_output

        per_frame_loss = torch.nn.functional.mse_loss(
            flow_pred.float(), target_model.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        per_frame_loss = per_frame_loss * self.scheduler.training_weight(
            timestep_model_input
        ).unflatten(0, (batch_size, num_frame + num_pairs))

        current_mask = valid_mask.clone()
        if context_frames > 0:
            current_mask[:, :context_frames] = 0
        goal_mask = torch.ones(
            (batch_size, num_pairs), device=per_frame_loss.device, dtype=per_frame_loss.dtype
        )
        if self.lookahead_start_chunk > 0:
            goal_mask[:, :self.lookahead_start_chunk] = 0

        current_loss = per_frame_loss[:, :num_frame]
        goal_loss = per_frame_loss[:, num_frame:]
        current_loss_mean = (current_loss * current_mask).sum() / current_mask.sum().clamp(min=1.0)
        goal_loss_mean = (goal_loss * goal_mask).sum() / goal_mask.sum().clamp(min=1.0)
        loss = current_loss_mean + self.endpoint_loss_weight * goal_loss_mean

        future_hidden_loss = None
        future_hidden_valid = None
        if self.future_hidden_alignment_enabled:
            if future_hidden_prediction is None or future_hidden_target is None:
                raise RuntimeError("Future-hidden alignment tensors were not returned by the generator.")
            tokens_per_chunk = future_hidden_prediction.shape[2]
            if tokens_per_chunk % block_size != 0:
                raise ValueError(
                    f"Alignment token count {tokens_per_chunk} is not divisible by "
                    f"block_size={block_size}."
                )
            frame_seqlen = tokens_per_chunk // block_size
            target_frame_mask = valid_mask[
                :, block_size:block_size + num_pairs * block_size
            ]
            if target_frame_mask.shape[1] != num_pairs * block_size:
                raise ValueError("Future-hidden target mask does not cover every next chunk.")
            future_hidden_valid = target_frame_mask.unflatten(
                1, (num_pairs, block_size)
            ).repeat_interleave(frame_seqlen, dim=2)
            future_hidden_weights = self._future_hidden_token_weights(
                raw_video,
                num_frames=num_frame,
                num_pairs=num_pairs,
                block_size=block_size,
                frame_seqlen=frame_seqlen,
            )
            future_hidden_loss = masked_future_hidden_cosine_loss(
                future_hidden_prediction,
                future_hidden_target,
                future_hidden_valid,
                token_weights=future_hidden_weights,
            )
            loss = loss + self.future_hidden_alignment_weight * future_hidden_loss

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred[:, :num_frame].detach(),
            "lookahead_current_loss": current_loss_mean.detach(),
            "lookahead_goal_loss": goal_loss_mean.detach(),
            "lookahead_goal_loss_weight": torch.tensor(
                self.endpoint_loss_weight, device=loss.device, dtype=loss.dtype
            ),
        }
        if future_hidden_loss is not None:
            log_dict.update({
                "future_hidden_alignment_loss": future_hidden_loss.detach(),
                "future_hidden_alignment_weight": torch.tensor(
                    self.future_hidden_alignment_weight,
                    device=loss.device,
                    dtype=loss.dtype,
                ),
                "future_hidden_alignment_valid_ratio": (
                    future_hidden_valid.float().mean().detach()
                    if future_hidden_valid.numel() > 0
                    else torch.zeros((), device=loss.device)
                ),
            })
        return loss, log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        loss_mask_global_valid_count: torch.Tensor = None,
        global_step: int = None,
        raw_video: torch.Tensor = None,
        goal_frame_index: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - loss_mask: optional tensor of shape [B, F] with 1.0 for valid frames and 0.0 for padded frames.
                         Under Sequence Parallel this is already the local chunk.
            - loss_mask_global_valid_count: optional scalar tensor with the total valid count across all SP ranks.
                         When provided (SP mode), used as the denominator instead of the local loss_mask.sum().
            - global_step: current training step, used for error recycling delayed start.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        effective_goal_frame_index = self.lookahead_goal_frame_index
        if effective_goal_frame_index is None and goal_frame_index is not None:
            effective_goal_frame_index = int(goal_frame_index[0].item())
        goal_has_future_pair = not (
            self.lookahead_stop_when_goal_not_future
            and effective_goal_frame_index is not None
            and not goal_is_strictly_future(
                0,
                self.num_frame_per_block,
                effective_goal_frame_index,
            )
        )
        if (
            self.future_lookahead
            and self.lookahead_layout == "goal_frame"
            and image_or_video_shape[1] // self.num_frame_per_block > 1
            and goal_has_future_pair
        ):
            return self.goal_frame_lookahead_generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
                loss_mask=loss_mask,
                loss_mask_global_valid_count=loss_mask_global_valid_count,
                global_step=global_step,
                raw_video=raw_video,
                goal_frame_index=goal_frame_index,
            )
        if (
            self.future_lookahead
            and self.lookahead_layout == "endpoint_replicas"
            and image_or_video_shape[1] // self.num_frame_per_block > 1
        ):
            return self.future_lookahead_generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
                loss_mask=loss_mask,
                loss_mask_global_valid_count=loss_mask_global_valid_count,
                global_step=global_step,
            )

        batch_size, num_frame = image_or_video_shape[:2]

        noise = torch.randn_like(clean_latent)
        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep( #不同 chunk 可以有不同 timestep,同一 chunk 内 8 个 latent frames 共享 timestep
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        context_latent = (
            initial_latent
            if getattr(self.args, "i2v", False) and initial_latent is not None
            else None
        )
        context_frames = int(context_latent.shape[1]) if context_latent is not None else 0
        if context_frames > 0:
            if context_frames >= num_frame:
                raise ValueError(
                    f"initial_latent has {context_frames} frames but training clip has {num_frame}."
                )
            timestep[:, :context_frames] = 0

        # Step 2.5 & 3.5: Error recycling — clean_prob acts as a master switch.
        # When clean_prob fires, skip ALL error injection and use pristine input;
        # otherwise each injection type rolls its own probability and is gated
        # only by whether the corresponding buffer has any samples (SVI behavior).
        #
        # NOTE on rank-sync: random.random() below is INTENTIONALLY independent
        # across ranks. None of these decisions guard a collective call (we use
        # SVI's pattern of unconditional all_gather + local random replay in
        # Step 5), so per-rank divergence here only affects which slice of data
        # gets corrupted on which rank — perfectly safe under DP+SP, and matches
        # SVI's behavior exactly.
        er_latent_injected = False
        er_noise_injected = False
        er_injected = False
        er_use_clean = False
        er_ready = (
            self.error_buffer is not None
            and (global_step is None or global_step >= self.er_start_step)
        )
        if er_ready and self.er_clean_prob > 0 and random.random() < self.er_clean_prob:
            er_use_clean = True

        # Noise error injection (SVI's noise_prob): corrupt noise input.
        # training_target is then computed with the corrupted noise so the
        # model learns to predict a self-correcting velocity (SVI Eq. logic).
        noise_for_train = noise
        if (
            er_ready and not er_use_clean
            and self.er_noise_inject_prob > 0
            and not self.noise_error_buffer.is_empty()
            and random.random() < self.er_noise_inject_prob
        ):
            noise_for_train = self._inject_noise_error_buffer(
                noise, index, batch_size, num_frame
            )
            er_noise_injected = True

        # Latent error injection (SVI's latent_prob): corrupt clean_latent
        # before noising. training_target keeps pointing to ORIGINAL clean_latent.
        clean_latent_for_noise = clean_latent
        if (
            er_ready and not er_use_clean
            and self.er_latent_inject_prob > 0
            and not self.error_buffer.is_empty()
            and random.random() < self.er_latent_inject_prob
        ):
            clean_latent_for_noise = self._inject_latent_error_buffer(
                clean_latent, index, batch_size, num_frame
            )
            er_latent_injected = True

        noisy_latents = self.scheduler.add_noise(
            clean_latent_for_noise.flatten(0, 1),
            noise_for_train.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        ##########
        training_target = self.scheduler.training_target(clean_latent, noise_for_train, timestep)
        ##########
        if context_frames > 0:
            noisy_latents[:, :context_frames] = context_latent.to(
                device=noisy_latents.device,
                dtype=noisy_latents.dtype,
            )
            training_target[:, :context_frames] = 0. #每个 chunk 的前 context_frames 个 latent frames 是 image condition, 不计算 loss

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        # Step 3.5: Error recycling — inject sampled errors into clean prefix.
        # 2D mode: per-position (random timestep). 1D mode: SVI global sampling.
        if (
            er_ready and not er_use_clean
            and not self.error_buffer.is_empty()
            and random.random() < self.er_context_inject_prob
        ):
            clean_latent_aug = self._inject_error_buffer(
                clean_latent_aug, index, batch_size, num_frame
            )
            er_injected = True

        if context_frames > 0:
            clean_latent_aug = _overwrite_i2v_context(
                clean_latent_aug, context_latent, context_frames
            )
            if timestep_clean_aug is not None:
                timestep_clean_aug = _zero_i2v_context_timestep(
                    timestep_clean_aug, context_frames
                )

        # Compute loss. The optional third output is produced only during
        # teacher-forced self-alignment training; inference remains unchanged.
        generator_output = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None,
        )
        future_hidden_prediction = None
        future_hidden_target = None
        if self.future_hidden_alignment_enabled:
            if len(generator_output) != 3:
                raise RuntimeError("Future-hidden alignment expected generator hidden states.")
            flow_pred, x0_pred, auxiliary_prediction = generator_output
            if not isinstance(auxiliary_prediction, dict):
                raise RuntimeError("Future-hidden alignment expected an auxiliary dictionary.")
            future_hidden_prediction = auxiliary_prediction.get("future_hidden_prediction")
            future_hidden_target = auxiliary_prediction.get("future_hidden_target")
        else:
            flow_pred, x0_pred = generator_output
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        if context_frames > 0:
            if loss_mask is None:
                loss_mask = torch.ones(
                    (batch_size, num_frame),
                    device=loss.device,
                    dtype=loss.dtype,
                )
            loss_mask[:, :context_frames] = 0
        if loss_mask is not None:
            loss = loss * loss_mask
            valid_count = loss_mask_global_valid_count if loss_mask_global_valid_count is not None else loss_mask.sum()
            loss = loss.sum() / valid_count.clamp(min=1.0)
        else:
            loss = loss.mean()

        future_hidden_loss = None
        future_hidden_valid = None
        if self.future_hidden_alignment_enabled:
            if future_hidden_prediction is None or future_hidden_target is None:
                raise RuntimeError("Future-hidden alignment tensors were not returned by the generator.")
            num_pairs = future_hidden_prediction.shape[1]
            tokens_per_chunk = future_hidden_prediction.shape[2]
            block_size = self.num_frame_per_block
            if tokens_per_chunk % block_size != 0:
                raise ValueError(
                    f"Alignment token count {tokens_per_chunk} is not divisible by "
                    f"block_size={block_size}."
                )
            if loss_mask is None:
                valid_mask = torch.ones(
                    (batch_size, num_frame), device=flow_pred.device, dtype=flow_pred.dtype
                )
            else:
                valid_mask = loss_mask
            target_frame_mask = valid_mask[
                :, block_size:block_size + num_pairs * block_size
            ]
            if target_frame_mask.shape[1] != num_pairs * block_size:
                raise ValueError("Future-hidden target mask does not cover every next chunk.")
            frame_seqlen = tokens_per_chunk // block_size
            future_hidden_valid = target_frame_mask.unflatten(
                1, (num_pairs, block_size)
            ).repeat_interleave(frame_seqlen, dim=2)
            future_hidden_weights = self._future_hidden_token_weights(
                raw_video,
                num_frames=num_frame,
                num_pairs=num_pairs,
                block_size=block_size,
                frame_seqlen=frame_seqlen,
            )
            future_hidden_loss = masked_future_hidden_cosine_loss(
                future_hidden_prediction,
                future_hidden_target,
                future_hidden_valid,
                token_weights=future_hidden_weights,
            )
            loss = loss + self.future_hidden_alignment_weight * future_hidden_loss

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        if future_hidden_loss is not None:
            log_dict.update({
                "future_hidden_alignment_loss": future_hidden_loss.detach(),
                "future_hidden_alignment_weight": torch.tensor(
                    self.future_hidden_alignment_weight,
                    device=loss.device,
                    dtype=loss.dtype,
                ),
                "future_hidden_alignment_valid_ratio": (
                    future_hidden_valid.float().mean().detach()
                    if future_hidden_valid.numel() > 0
                    else torch.zeros((), device=loss.device)
                ),
            })

        # Step 5: Store prediction errors into error buffer.
        #
        # SVI-style two-phase pattern (avoids the rank-divergent collective
        # deadlock that an ``if random.random() < p: all_gather()`` would
        # introduce):
        #   PHASE A (collective, UNCONDITIONAL): every rank reaches the
        #     all_gather call together so NCCL stays in sync.
        #   PHASE B (local, GATED): each rank independently decides whether
        #     to actually replay the gathered items into its buffer. The
        #     gate uses random.random() per rank — divergence here is fine
        #     because no further collective follows.
        if self.error_buffer is not None:
            with torch.no_grad():
                # latent error: x0_pred - clean_latent ≡ -σ(v_pred - v_gt) — used for context/latent injection
                latent_err = x0_pred.detach() - clean_latent.detach()
                # noise error: SVI definition is (1-σ)(v_pred - v_gt) so the buffer entry,
                # when later added directly to noise, equals ε_pred - ε_gt.
                sigma = self.scheduler.sigmas.to(flow_pred.device)[index].reshape(
                    batch_size, num_frame, 1, 1, 1
                ).to(flow_pred.dtype)
                noise_err = (flow_pred.detach() - training_target.detach()) * (1.0 - sigma)

                use_distributed = (
                    global_step is not None
                    and global_step <= self.er_buffer_warmup_iter
                )

                # === PHASE A: collective — runs on EVERY rank, no gating ===
                if use_distributed:
                    lat_items = self._gather_errors_for_buffer(
                        self.error_buffer, latent_err, index, batch_size, num_frame
                    )
                    noise_items = self._gather_errors_for_buffer(
                        self.noise_error_buffer, noise_err, index, batch_size, num_frame
                    )
                else:
                    lat_items = self._collect_local_items(
                        self.error_buffer, latent_err, index, batch_size, num_frame
                    )
                    noise_items = self._collect_local_items(
                        self.noise_error_buffer, noise_err, index, batch_size, num_frame
                    )

                # === PHASE B: local replay — random.random() per-rank is OK ===
                # When the input was clean (low-error), only update buffer with
                # small probability to avoid flooding it with near-zero samples
                # (SVI: clean_buffer_update_prob).
                should_update = True
                if er_use_clean and random.random() >= self.er_clean_buffer_update_prob:
                    should_update = False
                if should_update:
                    self._apply_gathered_items(self.error_buffer, lat_items)
                    self._apply_gathered_items(self.noise_error_buffer, noise_items)
            buf_stats = self.error_buffer.stats()
            noise_buf_stats = self.noise_error_buffer.stats()
            log_dict["er_total_added"] = buf_stats["total_added"]
            log_dict["er_filled_buckets"] = buf_stats["filled_buckets"]
            log_dict["er_total_entries"] = buf_stats["total_entries"]
            log_dict["er_noise_total_entries"] = noise_buf_stats["total_entries"]
            log_dict["er_injected"] = er_injected
            log_dict["er_latent_injected"] = er_latent_injected
            log_dict["er_noise_injected"] = er_noise_injected

        return loss, log_dict


    def _inject_error_buffer(self, clean_latent_aug, index, batch_size, num_frame):
        """Inject errors into the clean prefix (E_img).

        2D (position-bucketed): the i-th LOCAL prefix block draws from
        ``buckets[(i, *)]`` with a RANDOM timestep — the clean prefix is
        the product of full ODE integration so its accumulated error can
        come from any noise level, but its magnitude scales with the
        block's global position. Note ``skip_block_0`` is interpreted in
        the GLOBAL frame: only the very first SP rank may skip its block 0.

        1D (timestep-bucketed): falls back to SVI ``sample_global``.
        """
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        result = clean_latent_aug.clone()
        for b in range(batch_size):
            for blk in range(num_blocks):
                if self.er_skip_block_0 and (self.er_block_offset + blk) == 0:
                    continue
                if self.er_num_blocks > 0:
                    err = self.error_buffer.sample_pos_any_t(
                        blk, device=result.device, dtype=result.dtype
                    )
                else:
                    err = self.error_buffer.sample_global(
                        device=result.device, dtype=result.dtype
                    )
                if err is not None:
                    start = blk * block_size
                    end = start + block_size
                    result[b, start:end] = result[b, start:end] + err
        return result

    def _inject_latent_error_buffer(self, clean_latent, index, batch_size, num_frame):
        """Inject errors into clean_latent before noising (E_vid).

        Matches BOTH block_position (LOCAL) and timestep when the buffer is
        2D, else only timestep (SVI default).
        """
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        index_per_block = index[:, ::block_size]
        result = clean_latent.clone()
        for b in range(batch_size):
            for blk in range(num_blocks):
                t_idx = index_per_block[b, blk].item()
                pos = blk if self.er_num_blocks > 0 else None
                err = self.error_buffer.sample(
                    t_idx, device=result.device, dtype=result.dtype,
                    block_pos=pos,
                )
                if err is not None:
                    start = blk * block_size
                    end = start + block_size
                    result[b, start:end] = result[b, start:end] + err
        return result

    def _inject_noise_error_buffer(self, noise, index, batch_size, num_frame):
        """Inject errors into the noise (E_noise).

        Same matching strategy as ``_inject_latent_error_buffer`` but reads
        from the dedicated noise buffer.
        """
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        index_per_block = index[:, ::block_size]
        result = noise.clone()
        for b in range(batch_size):
            for blk in range(num_blocks):
                t_idx = index_per_block[b, blk].item()
                pos = blk if self.er_num_blocks > 0 else None
                err = self.noise_error_buffer.sample(
                    t_idx, device=result.device, dtype=result.dtype,
                    block_pos=pos,
                )
                if err is not None:
                    start = blk * block_size
                    end = start + block_size
                    result[b, start:end] = result[b, start:end] + err
        return result

    def _gather_errors_for_buffer(
        self, buffer, error, index, batch_size, num_frame
    ):
        """All-gather errors/timesteps across the appropriate group and return
        a list of ready-to-add ``(err_block, t_idx, pos_or_None)`` items.

        ★ This is a COLLECTIVE — every rank MUST reach this call together.
        The caller is responsible for invoking it unconditionally during the
        warmup window (just like SVI's ``all_gather`` outside the random
        ``if`` blocks). Random decisions about whether to actually consume
        the returned items belong to ``_apply_gathered_items`` instead.

        Group selection mirrors SVI's intent:
          * **2D (num_blocks > 0)** — DP group only. Other SP ranks' samples
            map to positions unreachable by this rank, so cross-SP gather
            wastes bandwidth.
          * **1D (num_blocks == 0)** — WORLD group (SVI default). Buckets
            are pos-agnostic so every rank's errors are valid samples.
        """
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return self._collect_local_items(buffer, error, index, batch_size, num_frame)

        if buffer.num_blocks > 0:
            from wan_5b.distributed.sp_training import get_data_parallel_group
            comm_group = get_data_parallel_group()
            if comm_group is None:
                return self._collect_local_items(buffer, error, index, batch_size, num_frame)
            comm_size = dist.get_world_size(comm_group)
        else:
            comm_group = None
            comm_size = dist.get_world_size()

        if comm_size <= 1:
            return self._collect_local_items(buffer, error, index, batch_size, num_frame)

        err_local = error.detach().contiguous()
        idx_local = index.detach().contiguous()
        err_list = [torch.empty_like(err_local) for _ in range(comm_size)]
        idx_list = [torch.empty_like(idx_local) for _ in range(comm_size)]
        if comm_group is None:
            dist.all_gather(err_list, err_local)
            dist.all_gather(idx_list, idx_local)
        else:
            dist.all_gather(err_list, err_local, group=comm_group)
            dist.all_gather(idx_list, idx_local, group=comm_group)

        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        items = []
        for err_r, idx_r in zip(err_list, idx_list):
            idx_per_block = idx_r[:, ::block_size]
            err_blocks = err_r.reshape(
                batch_size, num_blocks, block_size, *err_r.shape[2:]
            )
            for b in range(batch_size):
                for blk in range(num_blocks):
                    pos = blk if buffer.num_blocks > 0 else None
                    items.append((err_blocks[b, blk], idx_per_block[b, blk].item(), pos))
        return items

    def _collect_local_items(self, buffer, error, index, batch_size, num_frame):
        """Same item-list format as ``_gather_errors_for_buffer`` but with no
        collective — used outside the warmup window or when distributed is off."""
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        idx_per_block = index[:, ::block_size]
        error_blocks = error.reshape(
            batch_size, num_blocks, block_size, *error.shape[2:]
        )
        items = []
        for b in range(batch_size):
            for blk in range(num_blocks):
                pos = blk if buffer.num_blocks > 0 else None
                items.append((error_blocks[b, blk], idx_per_block[b, blk].item(), pos))
        return items

    def _apply_gathered_items(self, buffer, items):
        """Pure local: drop ``items`` into ``buffer``. No collective, no
        cross-rank coordination — each rank may invoke this independently
        (or skip it entirely) without risking a deadlock."""
        for err_block, t_idx, pos in items:
            buffer.add(err_block, t_idx, block_pos=pos)

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = CausalDiffusionInferencePipeline(
            args=self.args,
            device=self.device,
            generator=self.generator,
            text_encoder=self.text_encoder,
            vae=self.vae
        )
