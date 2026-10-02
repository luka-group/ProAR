#!/usr/bin/env python3
"""Convert FlowWAM WorldArena HDF5 episodes to this repo's video dataset.

The generated layout is compatible with ``MultiVideoConcatDataset``::

    OUTPUT_ROOT/
      video/T00_adjust_bottle__episode_0000000/0.mp4
      caption/T00_adjust_bottle__episode_0000000/0.json

Each output video contains a uniform temporal sample of the complete episode.
Frame 0 comes from the paired low-resolution episode and is bicubic-upsampled;
the remaining frames come from the high-resolution episode.  Consequently the
last output frame is always the episode's true final frame.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Episode:
    task_index: int
    task: str
    episode_name: str
    high_path: Path
    low_path: Path
    instruction_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FlowWAM WorldArena episodes for AR video training."
    )
    parser.add_argument("--high-res-root", type=Path, required=True)
    parser.add_argument("--low-res-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / "data" / "worldarena-train",
    )
    parser.add_argument("--variant", default="aloha-agilex_clean_50")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=50,
        help="Use the first N numbered episodes per task; 50 uses all clean_50 data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode_?(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse episode number from {path.name}")
    return int(match.group(1))


def discover_episodes(args: argparse.Namespace) -> list[Episode]:
    tasks = sorted(
        task.name
        for task in args.high_res_root.iterdir()
        if task.is_dir() and (task / args.variant / "data").is_dir()
    )
    episodes: list[Episode] = []
    for task_index, task in enumerate(tasks):
        high_variant = args.high_res_root / task / args.variant
        low_variant = args.low_res_root / task / args.variant
        for high_path in sorted(
            (high_variant / "data").glob("episode*.hdf5"), key=episode_number
        ):
            if episode_number(high_path) >= args.episodes_per_task:
                continue
            low_path = low_variant / "data" / high_path.name
            instruction_path = (
                high_variant / "instructions" / f"{high_path.stem}.json"
            )
            if not low_path.is_file():
                raise FileNotFoundError(f"Missing paired low-resolution episode: {low_path}")
            if not instruction_path.is_file():
                raise FileNotFoundError(f"Missing instruction: {instruction_path}")
            episodes.append(
                Episode(
                    task_index=task_index,
                    task=task,
                    episode_name=high_path.stem,
                    high_path=high_path,
                    low_path=low_path,
                    instruction_path=instruction_path,
                )
            )
    if not episodes:
        raise ValueError(f"No episodes found under {args.high_res_root}")
    return episodes


def decode_hdf5_image(value) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if isinstance(value, np.ndarray):
        encoded = np.frombuffer(value.tobytes(), dtype=np.uint8)
    elif not isinstance(value, (bytes, bytearray)):
        encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    else:
        encoded = np.frombuffer(value, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode HDF5 RGB frame")
    return image


def load_caption(path: Path, task: str) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("seen", "instruction", "instructions", "unseen"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return task.replace("_", " ")


def uniform_indices(length: int, num_frames: int) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"Episode has invalid length {length}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames == 1:
        return np.array([0], dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, num_frames)).astype(np.int64)


def initialize_worker() -> None:
    # OpenCV's internal thread pool plus multiple Python workers can oversubscribe
    # the CPU and has deadlocked with fork on some clusters.
    cv2.setNumThreads(1)


def process_episode(
    episode: Episode,
    output_root: Path,
    camera: str,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
    overwrite: bool,
) -> dict:
    sample_name = (
        f"T{episode.task_index:02d}_{episode.task}__{episode.episode_name}"
    )
    video_dir = output_root / "video" / sample_name
    caption_dir = output_root / "caption" / sample_name
    video_path = video_dir / "0.mp4"
    caption_path = caption_dir / "0.json"

    if video_path.is_file() and caption_path.is_file() and not overwrite:
        return {"sample": sample_name, "status": "skipped"}

    video_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    temporary_video_path = video_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {temporary_video_path}")
    try:
        dataset_key = f"observation/{camera}/rgb"
        with h5py.File(episode.high_path, "r") as high_file, h5py.File(
            episode.low_path, "r"
        ) as low_file:
            if dataset_key not in high_file or dataset_key not in low_file:
                raise KeyError(
                    f"Missing {dataset_key} in {episode.high_path} or {episode.low_path}"
                )
            high_rgb = high_file[dataset_key]
            low_rgb = low_file[dataset_key]
            if len(high_rgb) != len(low_rgb):
                raise ValueError(
                    f"Resolution pair length mismatch for {episode.task}/{episode.episode_name}: "
                    f"high={len(high_rgb)}, low={len(low_rgb)}"
                )
            source_length = len(high_rgb)
            indices = uniform_indices(source_length, num_frames)
            for output_index, source_index in enumerate(indices):
                source = low_rgb if output_index == 0 else high_rgb
                frame = decode_hdf5_image(source[int(source_index)])
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(
                        frame, (width, height), interpolation=cv2.INTER_CUBIC
                    )
                writer.write(frame)
    finally:
        writer.release()
    os.replace(temporary_video_path, video_path)

    caption = load_caption(episode.instruction_path, episode.task)
    temporary_caption_path = caption_path.with_suffix(".tmp.json")
    with temporary_caption_path.open("w", encoding="utf-8") as handle:
        json.dump({"caption": caption}, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_caption_path, caption_path)

    return {
        "sample": sample_name,
        "status": "written",
        "source_frames": source_length,
        "output_frames": len(indices),
    }


def main() -> None:
    args = parse_args()
    episodes = discover_episodes(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Preparing {len(episodes)} episodes at "
        f"{args.width}x{args.height}, {args.num_frames} frames, {args.fps:g} fps"
    )

    written = 0
    skipped = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
    ) as executor:
        futures = [
            executor.submit(
                process_episode,
                episode,
                args.output_root,
                args.camera,
                args.num_frames,
                args.width,
                args.height,
                args.fps,
                args.overwrite,
            )
            for episode in episodes
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            written += result["status"] == "written"
            skipped += result["status"] == "skipped"
            if completed % 10 == 0 or completed == len(futures):
                print(
                    f"[{completed}/{len(futures)}] written={written}, skipped={skipped}",
                    flush=True,
                )

    print(f"Done: written={written}, skipped={skipped}, output={args.output_root}")


if __name__ == "__main__":
    main()
