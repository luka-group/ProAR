#!/usr/bin/env python3
"""Convert the extracted WorldArena test package to MultiVideoConcatDataset layout."""

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
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TestEpisode:
    number: int
    video_path: Path
    first_frame_path: Path
    instruction_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / "data" / "worldarena-test",
    )
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse episode number from {path.name}")
    return int(match.group(1))


def discover_episodes(root: Path) -> list[TestEpisode]:
    video_root = root / "video"
    first_frame_root = root / "first_frame" / "fixed_scene_task"
    instruction_root = root / "instructions" / "fixed_scene_task"
    episodes: list[TestEpisode] = []
    for video_path in sorted(video_root.glob("episode*.mp4"), key=episode_number):
        number = episode_number(video_path)
        first_frame_path = first_frame_root / f"episode{number}.png"
        instruction_path = instruction_root / f"episode{number}.json"
        if not first_frame_path.is_file():
            raise FileNotFoundError(f"Missing first frame: {first_frame_path}")
        if not instruction_path.is_file():
            raise FileNotFoundError(f"Missing instruction: {instruction_path}")
        episodes.append(TestEpisode(number, video_path, first_frame_path, instruction_path))
    if not episodes:
        raise ValueError(f"No test videos found under {video_root}")
    return episodes


def uniform_indices(length: int, num_frames: int) -> np.ndarray:
    if length <= 0 or num_frames <= 0:
        raise ValueError(f"Invalid sampling request: length={length}, frames={num_frames}")
    if num_frames == 1:
        return np.array([0], dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, num_frames)).astype(np.int64)


def initialize_worker() -> None:
    cv2.setNumThreads(1)


def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)


def process_episode(episode: TestEpisode, args: argparse.Namespace) -> dict:
    # The public test package has no reliable task label. A unique prefix lets
    # periodic evaluation sample distinct official test episodes without
    # pretending that an inferred task taxonomy is ground truth.
    sample_name = f"E{episode.number:04d}__episode{episode.number}"
    video_dir = args.output_root / "video" / sample_name
    caption_dir = args.output_root / "caption" / sample_name
    video_path = video_dir / "0.mp4"
    caption_path = caption_dir / "0.json"
    if video_path.is_file() and caption_path.is_file() and not args.overwrite:
        return {"status": "skipped", "sample": sample_name}

    capture = cv2.VideoCapture(str(episode.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {episode.video_path}")
    source_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = uniform_indices(source_frames, args.num_frames)

    first_frame = cv2.imread(str(episode.first_frame_path), cv2.IMREAD_COLOR)
    if first_frame is None:
        capture.release()
        raise RuntimeError(f"Cannot read {episode.first_frame_path}")
    first_frame = resize(first_frame, args.width, args.height)

    video_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    temporary_video = video_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create {temporary_video}")

    cached_index = None
    cached_frame = None
    try:
        for output_index, source_index in enumerate(indices):
            if output_index == 0:
                frame = first_frame
            elif cached_index == int(source_index):
                frame = cached_frame
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Failed reading frame {source_index} from {episode.video_path}"
                    )
                frame = resize(frame, args.width, args.height)
                cached_index = int(source_index)
                cached_frame = frame
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    os.replace(temporary_video, video_path)

    instruction = json.loads(episode.instruction_path.read_text(encoding="utf-8"))
    caption = str(instruction.get("instruction", "")).strip()
    if not caption:
        raise ValueError(f"Empty instruction: {episode.instruction_path}")
    temporary_caption = caption_path.with_suffix(".tmp.json")
    temporary_caption.write_text(
        json.dumps({"caption": caption}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_caption, caption_path)
    return {
        "status": "written",
        "sample": sample_name,
        "source_frames": source_frames,
        "output_frames": len(indices),
    }


def main() -> None:
    args = parse_args()
    episodes = discover_episodes(args.input_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {len(episodes)} WorldArena test episodes", flush=True)
    written = skipped = 0
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
    ) as executor:
        futures = [executor.submit(process_episode, episode, args) for episode in episodes]
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
