#!/usr/bin/env python3
"""Run dataset evaluation from a diffusion checkpoint without training.

Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    scripts/evaluate_diffusion_checkpoint.py \
    --config_path configs/train_ar_videorlvr_maze.yaml \
    --checkpoint outputs/ar_videorlvr_maze/checkpoint_model_010000/model.pt \
    --output_dir outputs/ar_videorlvr_maze/test_10000 \
    --max_samples 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="Override the evaluation dataset path from the config.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optionally evaluate only these task prefixes, for example O-47.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Override evaluation generation length in latent frames.",
    )
    parser.add_argument(
        "--output-pixel-frames",
        type=int,
        default=None,
        help="Keep only this many decoded pixel frames when saving each video.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help="Override the number of diffusion denoising steps per generated chunk.",
    )
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-save-dir", default="")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-upload-interval", type=int, default=20)
    parser.add_argument(
        "--data-parallel-eval",
        action="store_true",
        help=(
            "Replicate the full model on every GPU and shard evaluation samples "
            "across ranks. Only rank 0 owns the W&B run and uploads all videos."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the evaluation RNG seed from the config.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--vbvr-evalkit-layout",
        action="store_true",
        help=(
            "Save videos as In-Domain_50/<task>/<sample>.mp4 and "
            "Out-of-Domain_50/<task>/<sample>.mp4 for VBVR-EvalKit."
        ),
    )
    parser.add_argument(
        "--physics-iq-layout",
        action="store_true",
        help=(
            "Save videos with their Physics-IQ sample names (0001_...mp4 through "
            "0198_...mp4). The evaluation dataset folder names must use "
            "<task>__<Physics-IQ generated_video_name stem>."
        ),
    )
    parser.add_argument(
        "--vbvr-gt-base",
        default=None,
        help="Optional official VBVR-Bench ground-truth root used to infer In/Out-Domain split.",
    )
    parser.add_argument(
        "--vbvr-split",
        choices=("all", "In-Domain_50", "Out-of-Domain_50"),
        default="all",
        help="When using --vbvr-evalkit-layout, generate only one VBVR split or all splits.",
    )
    return parser.parse_args()


def checkpoint_step(checkpoint: str) -> int | None:
    parent = Path(checkpoint).parent.name
    if not parent.startswith("checkpoint_model_"):
        return None
    try:
        return int(parent.replace("checkpoint_model_", "", 1))
    except ValueError:
        return None


def load_video_metadata(metadata_path: Path) -> dict | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read {metadata_path}: {exc}", flush=True)
        return None
    video_path = Path(metadata.get("video_path", str(metadata_path.with_suffix(".mp4"))))
    if not video_path.exists():
        print(f"Warning: missing generated video: {video_path}", flush=True)
        return None
    metadata["video_path"] = str(video_path)
    return metadata


def build_vbvr_split_map(eval_data_path: str | Path) -> dict[tuple[str, str], str]:
    """Recover VBVR split from LongLive metadata when the official GT root is absent."""
    split_map: dict[tuple[str, str], str] = {}
    metadata_root = Path(eval_data_path) / "metadata"
    if not metadata_root.is_dir():
        return split_map

    for metadata_path in sorted(metadata_root.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_name = metadata.get("task") or metadata.get("task_name")
        sample_id = metadata.get("sample_id")
        if not task_name or sample_id is None:
            continue
        source = f"{metadata.get('source_dir', '')} {metadata.get('source_video', '')}"
        if "In-Domain_50" in source:
            split_map[(str(task_name), str(sample_id))] = "In-Domain_50"
        elif "Out-of-Domain_50" in source:
            split_map[(str(task_name), str(sample_id))] = "Out-of-Domain_50"
    return split_map


def infer_vbvr_split(
    task_name: str,
    sample_id: str,
    *,
    gt_base: Path | None,
    split_map: dict[tuple[str, str], str],
) -> str | None:
    if gt_base is not None:
        for split in ("In-Domain_50", "Out-of-Domain_50"):
            if (gt_base / split / task_name / sample_id).is_dir():
                return split
    return split_map.get((task_name, sample_id))


def output_paths_for_sample(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    rank: int,
    sample_idx: int,
    task_prefix: str,
    task_name: str,
    sample_id: str,
    vbvr_split: str | None,
) -> tuple[Path, Path, Path]:
    if args.physics_iq_layout:
        if not re.match(r"^\d{4}_", sample_id):
            raise ValueError(
                "Physics-IQ sample_id must start with a zero-padded four-digit ID; "
                f"got {sample_id!r}."
            )
        video_path = output_dir / f"{sample_id}.mp4"
        metadata_path = output_dir / "_metadata" / f"{sample_id}.json"
        prompt_path = output_dir / "_prompts" / f"{sample_id}.txt"
        return video_path, metadata_path, prompt_path

    if args.vbvr_evalkit_layout:
        if vbvr_split is None:
            raise ValueError("VBVR EvalKit layout requires a resolved split.")
        video_path = output_dir / vbvr_split / task_name / f"{sample_id}.mp4"
        metadata_path = output_dir / "_metadata" / vbvr_split / task_name / f"{sample_id}.json"
        prompt_path = output_dir / "_prompts" / vbvr_split / task_name / f"{sample_id}.txt"
        return video_path, metadata_path, prompt_path

    video_stem = f"video_{task_prefix}_rank{rank:02d}_idx{sample_idx:06d}"
    video_path = output_dir / f"{video_stem}.mp4"
    metadata_path = output_dir / f"{video_stem}.json"
    prompt_path = output_dir / f"prompt_rank{rank:02d}_idx{sample_idx:06d}.txt"
    return video_path, metadata_path, prompt_path


def write_generation_metadata(
    metadata_path: Path,
    *,
    step: int | None,
    rank: int,
    task_prefix: str,
    task_name: str,
    sample_id: str,
    sample_idx: int,
    caption: str,
    video_path: Path,
    generation_latent_frames: int | None,
    saved_pixel_frames: int | None,
    vbvr_split: str | None,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "rank": rank,
        "task_prefix": task_prefix,
        "task_name": task_name,
        "sample_id": sample_id,
        "sample_idx": sample_idx,
        "mode": "model",
        "caption": caption,
        "video_path": str(video_path),
        "generation_latent_frames": generation_latent_frames,
        "saved_pixel_frames": saved_pixel_frames,
    }
    if vbvr_split is not None:
        payload["split"] = vbvr_split
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def wandb_log_video_batch(wandb, metadata_paths: list[Path], *, fps: int, step: int, prefix: str) -> int:
    if not metadata_paths:
        return 0
    table = wandb.Table(
        columns=[
            "step", "split", "task_prefix", "task_name", "sample_id",
            "sample_idx", "rank", "video", "caption", "path",
        ]
    )
    media_log = {}
    logged = 0
    for metadata_path in sorted(metadata_paths):
        metadata = load_video_metadata(metadata_path)
        if metadata is None:
            continue
        sample_idx = int(metadata.get("sample_idx", 0))
        task_prefix = metadata.get("task_prefix", "unknown")
        split = metadata.get("split", "")
        caption = metadata.get("caption", "")
        video_path = metadata["video_path"]
        media_key = f"{prefix}/{split + '/' if split else ''}{task_prefix}/sample_{sample_idx:06d}"
        media_log[media_key] = wandb.Video(video_path, caption=caption, fps=fps, format="mp4")
        table.add_data(
            metadata.get("step", step),
            split,
            task_prefix,
            metadata.get("task_name", "unknown"),
            metadata.get("sample_id", str(sample_idx)),
            sample_idx,
            int(metadata.get("rank", 0)),
            wandb.Video(video_path, caption=caption, fps=fps, format="mp4"),
            caption,
            video_path,
        )
        logged += 1
    if logged == 0:
        return 0
    payload = {
        f"{prefix}_batch": table,
        f"{prefix}_logged_total": logged,
    }
    payload.update(media_log)
    wandb.log(payload, step=step)
    print(f"[wandb] uploaded {logged} videos", flush=True)
    return logged


def main() -> None:
    args = parse_args()

    if args.physics_iq_layout and args.vbvr_evalkit_layout:
        raise ValueError("--physics-iq-layout and --vbvr-evalkit-layout are mutually exclusive.")

    import torchvision.io as _tv_io
    from torchvision.io import write_video

    if not hasattr(_tv_io, "write_video"):
        import imageio.v2 as _imageio_v2

        def _shim_write_video(filename, video_array, fps, **_unused):
            if hasattr(video_array, "detach"):
                video_array = video_array.detach().cpu().numpy()
            _imageio_v2.mimwrite(filename, video_array, fps=fps, codec="libx264", quality=8)

        _tv_io.write_video = _shim_write_video

    import torch.distributed as dist
    import torch
    import wandb
    from omegaconf import OmegaConf

    from trainer import DiffusionTrainer
    from trainer.diffusion import save_prompts_to_txt
    from utils.config import normalize_config, section_get
    from utils.distributed import barrier

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = Path(args.output_dir)
    step = checkpoint_step(str(checkpoint))

    rank_env = int(os.environ.get("RANK", "0"))
    if rank_env == 0:
        if output_dir.exists():
            if args.overwrite:
                shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    config = normalize_config(OmegaConf.load(args.config_path))
    config.config_name = Path(args.config_path).stem + "_eval"
    config.generator_ckpt = str(checkpoint)
    config.logdir = str(output_dir)
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.wandb_run_name:
        config.config_name = args.wandb_run_name
    config.auto_resume = False
    config.no_save = True
    config.no_visualize = False
    config.eval_only = True
    config.generate_before_train = False
    if args.data_parallel_eval:
        if int(config.sequence_parallel_size) != 1:
            raise ValueError("--data-parallel-eval requires sequence_parallel_size=1")
        config.sharding_strategy = "no_shard"
        if config.get("infra", None) is not None:
            config.infra.sharding_strategy = "no_shard"
    if args.eval_data_path:
        config.eval_data_path = args.eval_data_path
        if config.get("data", None) is not None:
            config.data.eval_data_path = args.eval_data_path
    if args.num_frames is not None:
        if args.num_frames <= 0:
            raise ValueError("--num-frames must be positive")
        config.evaluation.num_frames = int(args.num_frames)
        config.evaluation.max_num_frames = int(args.num_frames)
        config.evaluation.match_gt_length = False
    if args.output_pixel_frames is not None:
        if args.output_pixel_frames <= 0:
            raise ValueError("--output-pixel-frames must be positive")
        config.evaluation.output_pixel_frames = int(args.output_pixel_frames)
    if args.sampling_steps is not None:
        if args.sampling_steps <= 0:
            raise ValueError("--sampling-steps must be positive")
        config.inference.sampling_steps = int(args.sampling_steps)
    # Full-checkpoint evaluation should not inherit the small per-task subset
    # used for periodic training visualizations.
    config.evaluation.samples_per_task = 0
    config.evaluation.random_task_count = 0
    config.samples_per_task = 0
    config.random_task_count = 0
    config.evaluation.max_samples = int(args.max_samples)
    config.max_samples = int(args.max_samples)

    vbvr_gt_base = Path(args.vbvr_gt_base) if args.vbvr_gt_base else None
    if vbvr_gt_base is not None and not vbvr_gt_base.exists():
        raise FileNotFoundError(f"VBVR ground-truth root not found: {vbvr_gt_base}")
    vbvr_split_map = build_vbvr_split_map(config.eval_data_path) if args.vbvr_evalkit_layout else {}
    if args.vbvr_evalkit_layout and vbvr_gt_base is None and not vbvr_split_map:
        raise ValueError(
            "--vbvr-evalkit-layout needs either --vbvr-gt-base or LongLive metadata "
            "containing In-Domain_50/Out-of-Domain_50 source paths."
        )

    trainer = DiffusionTrainer(config)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    selected_tasks = set(args.tasks or [])
    if selected_tasks:
        available_tasks = set(getattr(trainer, "eval_task_prefixes", []))
        missing_tasks = selected_tasks - available_tasks
        if missing_tasks:
            raise ValueError(
                f"Requested tasks are absent from the evaluation dataset: "
                f"{sorted(missing_tasks)}. Available tasks: {sorted(available_tasks)}."
            )

    if trainer.model.inference_pipeline is None:
        trainer.model._initialize_inference_pipeline()

    max_samples = int(args.max_samples)
    eligible_seen = 0
    considered = 0
    generated = 0
    skipped = 0
    wandb_pending: list[Path] = []
    wandb_uploaded = 0
    wandb_upload_batches = 0
    wandb_interval = max(1, int(args.wandb_upload_interval))
    save_latents_only = bool(config.get("save_latents_only", False))
    if save_latents_only:
        raise ValueError("This evaluation script is for video export; set save_latents_only=false.")

    def maybe_upload_pending(force: bool = False) -> None:
        nonlocal wandb_uploaded, wandb_upload_batches
        if args.disable_wandb or rank != 0:
            return
        if not force and len(wandb_pending) < wandb_interval:
            return
        batch = list(wandb_pending)
        wandb_pending.clear()
        wandb_upload_batches += 1
        log_step = (step or 0) + wandb_upload_batches
        uploaded = wandb_log_video_batch(
            wandb,
            batch,
            fps=trainer.fps,
            step=log_step,
            prefix="test_videos",
        )
        wandb_uploaded += uploaded
        wandb.log(
            {
                "test_progress/considered": considered,
                "test_progress/generated": generated,
                "test_progress/skipped_existing": skipped,
                "test_progress/wandb_uploaded": wandb_uploaded,
                "test_progress/upload_batches": wandb_upload_batches,
            },
            step=log_step,
        )

    for eval_batch in trainer.eval_dataloader:
        eval_prompts = eval_batch["prompts"]
        eval_idx = eval_batch["idx"]
        eval_images = eval_batch.get("image", None)
        eval_num_latent_frames = eval_batch.get("eval_num_latent_frames", None)
        eval_save_frames = eval_batch.get("eval_save_frames", None)
        eval_num_valid_latent_frames = eval_batch.get("num_valid_latent_frames", None)
        eval_goal_frame_index = eval_batch.get("goal_frame_index", None)
        eval_task_prefixes = eval_batch.get("task_prefix", None)
        eval_task_names = eval_batch.get("task_name", None)
        eval_sample_ids = eval_batch.get("sample_id", None)

        for b in range(len(eval_prompts)):
            prompts_for_sample = eval_prompts[b]
            sample_idx = (
                eval_idx[b].item()
                if hasattr(eval_idx, "shape")
                else int(eval_idx[b])
            )
            task_prefix = eval_task_prefixes[b] if eval_task_prefixes else "unknown"
            task_name = eval_task_names[b] if eval_task_names else "unknown"
            sample_id = eval_sample_ids[b] if eval_sample_ids else str(sample_idx)
            if selected_tasks and task_prefix not in selected_tasks:
                continue
            vbvr_split = None
            if args.vbvr_evalkit_layout:
                vbvr_split = infer_vbvr_split(
                    task_name,
                    sample_id,
                    gt_base=vbvr_gt_base,
                    split_map=vbvr_split_map,
                )
                if vbvr_split is None:
                    raise RuntimeError(
                        f"Could not infer VBVR split for {task_name}/{sample_id}. "
                        "Pass --vbvr-gt-base pointing to the official VBVR-Bench data."
                    )
                if args.vbvr_split != "all" and vbvr_split != args.vbvr_split:
                    continue

            if max_samples > 0 and eligible_seen >= max_samples:
                break
            sample_position = eligible_seen
            eligible_seen += 1
            if args.data_parallel_eval and sample_position % world_size != rank:
                continue
            considered += 1

            video_path, metadata_path, prompt_path = output_paths_for_sample(
                args=args,
                output_dir=output_dir,
                # Sample indices are globally unique in data-parallel mode. In
                # FSDP mode all ranks jointly generate the same sample.
                rank=0,
                sample_idx=sample_idx,
                task_prefix=task_prefix,
                task_name=task_name,
                sample_id=sample_id,
                vbvr_split=vbvr_split,
            )

            generation_latent_frames = (
                int(eval_num_latent_frames[b].item())
                if eval_num_latent_frames is not None else None
            )
            saved_pixel_frames = (
                int(eval_save_frames[b].item())
                if eval_save_frames is not None else None
            )

            if args.data_parallel_eval:
                video_exists = video_path.exists()
            else:
                exists_flag = torch.tensor(
                    [int(video_path.exists()) if rank == 0 else 0],
                    device=f"cuda:{torch.cuda.current_device()}",
                    dtype=torch.int32,
                )
                dist.broadcast(exists_flag, src=0)
                video_exists = bool(exists_flag.item())
            if video_exists:
                if (args.data_parallel_eval or rank == 0) and not metadata_path.exists():
                    write_generation_metadata(
                        metadata_path,
                        step=step,
                        rank=rank if args.data_parallel_eval else 0,
                        task_prefix=task_prefix,
                        task_name=task_name,
                        sample_id=sample_id,
                        sample_idx=sample_idx,
                        caption=prompts_for_sample[0] if len(prompts_for_sample) > 0 else "",
                        video_path=video_path,
                        generation_latent_frames=generation_latent_frames,
                        saved_pixel_frames=saved_pixel_frames,
                        vbvr_split=vbvr_split,
                    )
                skipped += 1
                if rank == 0 and (skipped <= 5 or skipped % 50 == 0):
                    print(f"[skip] {video_path}", flush=True)
                if rank == 0 and not args.data_parallel_eval:
                    wandb_pending.append(metadata_path)
                    maybe_upload_pending()
                if not args.data_parallel_eval:
                    barrier()
                continue

            if args.data_parallel_eval or rank == 0:
                print(
                    f"[rank {rank}] [generate] {task_prefix}/{sample_id} idx={sample_idx}",
                    flush=True,
                )

            generated_video = trainer.generate_video(
                trainer.model.inference_pipeline,
                [prompts_for_sample],
                eval_images[b:b + 1] if eval_images is not None else None,
                use_ema=False,
                num_frames=generation_latent_frames,
                goal_frame_index=(
                    int(eval_goal_frame_index[b].item())
                    if eval_goal_frame_index is not None
                    else (
                        int(eval_num_valid_latent_frames[b].item()) - 1
                        if eval_num_valid_latent_frames is not None else None
                    )
                ),
            )
            video_to_save = generated_video[0]
            output_pixel_frames = int(section_get(
                trainer.config,
                "evaluation",
                "output_pixel_frames",
                saved_pixel_frames or 0,
            ) or 0)
            if output_pixel_frames > 0:
                if video_to_save.shape[0] < output_pixel_frames:
                    raise ValueError(
                        f"Generated {video_to_save.shape[0]} pixel frames, fewer than "
                        f"the requested {output_pixel_frames}."
                    )
                output_frame_sampling = str(section_get(
                    trainer.config,
                    "evaluation",
                    "output_frame_sampling",
                    "head",
                ))
                if output_frame_sampling == "uniform":
                    indices = [
                        round(i * (video_to_save.shape[0] - 1) / (output_pixel_frames - 1))
                        for i in range(output_pixel_frames)
                    ] if output_pixel_frames > 1 else [0]
                    video_to_save = video_to_save[indices]
                elif output_frame_sampling == "head":
                    video_to_save = video_to_save[:output_pixel_frames]
                else:
                    raise ValueError(
                        "evaluation.output_frame_sampling must be 'head' or 'uniform'."
                    )
            saved_pixel_frames = int(video_to_save.shape[0])
            if args.data_parallel_eval or rank == 0:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                write_video(str(video_path), video_to_save, fps=trainer.fps)
                write_generation_metadata(
                    metadata_path,
                    step=step,
                    rank=rank if args.data_parallel_eval else 0,
                    task_prefix=task_prefix,
                    task_name=task_name,
                    sample_id=sample_id,
                    sample_idx=sample_idx,
                    caption=prompts_for_sample[0] if len(prompts_for_sample) > 0 else "",
                    video_path=video_path,
                    generation_latent_frames=generation_latent_frames,
                    saved_pixel_frames=saved_pixel_frames,
                    vbvr_split=vbvr_split,
                )
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                save_prompts_to_txt(prompts_for_sample, str(prompt_path), trainer.is_main_process)
            generated += 1
            if rank == 0 and not args.data_parallel_eval:
                wandb_pending.append(metadata_path)
                maybe_upload_pending()
            del generated_video
            gc.collect()
            torch.cuda.empty_cache()
            if not args.data_parallel_eval:
                barrier()

        if max_samples > 0 and eligible_seen >= max_samples:
            break

    if args.data_parallel_eval:
        counts = torch.tensor(
            [considered, generated, skipped],
            device=f"cuda:{torch.cuda.current_device()}",
            dtype=torch.long,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        considered, generated, skipped = (int(value) for value in counts.tolist())

    print(
        f"[rank {rank}] considered={considered}, generated={generated}, skipped_existing={skipped}",
        flush=True,
    )
    barrier()

    if (not args.disable_wandb) and rank == 0:
        all_metadata_paths = sorted(
            (output_dir / "_metadata").glob("**/*.json")
            if args.vbvr_evalkit_layout or args.physics_iq_layout
            else output_dir.glob("video_*.json")
        )
        if args.data_parallel_eval:
            for batch_start in range(0, len(all_metadata_paths), wandb_interval):
                wandb_upload_batches += 1
                batch = all_metadata_paths[batch_start:batch_start + wandb_interval]
                wandb_uploaded += wandb_log_video_batch(
                    wandb,
                    batch,
                    fps=trainer.fps,
                    step=(step or 0) + wandb_upload_batches,
                    prefix="test_videos",
                )
        else:
            maybe_upload_pending(force=True)
        summary_table = wandb.Table(
            columns=["sample_idx", "split", "task_prefix", "task_name", "sample_id", "caption", "path"]
        )
        for metadata_path in all_metadata_paths:
            metadata = load_video_metadata(metadata_path)
            if metadata is None:
                continue
            summary_table.add_data(
                int(metadata.get("sample_idx", 0)),
                metadata.get("split", ""),
                metadata.get("task_prefix", "unknown"),
                metadata.get("task_name", "unknown"),
                metadata.get("sample_id", ""),
                metadata.get("caption", ""),
                metadata.get("video_path", ""),
            )
        wandb.log(
            {
                "test_videos_summary": summary_table,
                "test_progress/considered": considered,
                "test_progress/generated": generated,
                "test_progress/skipped_existing": skipped,
                "test_progress/wandb_uploaded": wandb_uploaded,
                "test_progress/upload_batches": wandb_upload_batches,
                "test_progress/local_video_count": len(all_metadata_paths),
            },
            step=(step or 0) + wandb_upload_batches + 1,
        )
        wandb.finish()
    elif not args.disable_wandb:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        print(f"Evaluation videos saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
