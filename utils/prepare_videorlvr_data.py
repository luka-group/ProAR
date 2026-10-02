#!/usr/bin/env python3
"""Download VideoRLVR and write the final three-task LongLive datasets.

Default output:

  data/videorlvr-datasets-3tasks/
    video/<task>__<sample_id>/0.mp4
    caption/<task>__<sample_id>/0.json
    metadata/<task>__<sample_id>.json
  data/videorlvr-test-3tasks/
    video/<task>__<sample_id>/0.mp4
    caption/<task>__<sample_id>/0.json
    metadata/<task>__<sample_id>.json

No per-task intermediate dataset, transcoding, or hardlink merge is required.
Interrupted runs can be resumed: a task is skipped only when its manifest and
expected output counts match the requested source revision.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_REPO_ID = "DarthZhu/VideoRLVR-Data"
DEFAULT_TASKS = ("maze", "flowfree", "sokoban")
SPLITS = ("train", "test")
EXPECTED_PER_TASK = {"train": 10_000, "test": 1_000}
OUTPUT_SUBDIRS = ("video", "caption", "metadata")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset branch, tag, or commit. The resolved commit SHA is saved in manifests.",
    )
    parser.add_argument(
        "--train-output-root",
        default=str(repo_root() / "data" / "videorlvr-datasets-3tasks"),
    )
    parser.add_argument(
        "--test-output-root",
        default=str(repo_root() / "data" / "videorlvr-test-3tasks"),
    )
    parser.add_argument(
        "--raw-root",
        default=str(repo_root() / "data" / ".cache" / "videorlvr-raw"),
        help="Temporary download directory.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=DEFAULT_TASKS)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use files already present under --raw-root.",
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild selected task outputs even when matching manifests exist.",
    )
    parser.add_argument(
        "--no-validate-counts",
        action="store_true",
        help="Do not enforce 10,000 train and 1,000 test samples per task.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root() / path


def split_output_root(args: argparse.Namespace, split: str) -> Path:
    if split == "train":
        return resolve_path(args.train_output_root)
    if split == "test":
        return resolve_path(args.test_output_root)
    raise ValueError(f"Unknown split: {split}")


def task_manifest_path(output_root: Path, task: str) -> Path:
    return output_root / "manifests" / f"{task}.json"


def resolve_revision(repo_id: str, revision: str, skip_download: bool) -> str:
    if skip_download:
        return revision
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it or use an environment containing the hf CLI."
        ) from exc
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve {repo_id}@{revision} to a commit SHA")
    return str(info.sha)


def task_is_complete(
    output_root: Path,
    *,
    repo_id: str,
    revision: str,
    split: str,
    task: str,
    expected_count: int | None,
) -> bool:
    manifest_path = task_manifest_path(output_root, task)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    count = int(manifest.get("total_samples", 0))
    if expected_count is not None and count != expected_count:
        return False
    if not all(
        (
            manifest.get("repo_id") == repo_id,
            manifest.get("revision") == revision,
            manifest.get("split") == split,
            manifest.get("task") == task,
            count > 0,
        )
    ):
        return False

    video_count = len(list((output_root / "video").glob(f"{task}__*/0.mp4")))
    caption_count = len(list((output_root / "caption").glob(f"{task}__*/0.json")))
    metadata_count = len(list((output_root / "metadata").glob(f"{task}__*.json")))
    return video_count == caption_count == metadata_count == count


def clean_task_output(output_root: Path, task: str) -> None:
    for subdir in ("video", "caption"):
        for path in (output_root / subdir).glob(f"{task}__*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for path in (output_root / "metadata").glob(f"{task}__*.json"):
        path.unlink()
    manifest_path = task_manifest_path(output_root, task)
    if manifest_path.exists():
        manifest_path.unlink()


def raw_task_dir(raw_root: Path, split: str, task: str) -> Path:
    return raw_root / split / task


def download_split_task(
    repo_id: str,
    revision: str,
    raw_dir: Path,
    split: str,
    task: str,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--local-dir",
            str(raw_dir),
            "--include",
            f"{split}/{task}/**",
        ]
    )


def read_metadata(raw_dir: Path, split: str, task: str) -> tuple[Path, list[dict[str, str]]]:
    task_dir = raw_dir / split / task
    metadata_candidates = [
        task_dir / "metadata.csv",
        raw_dir / split / f"metadata_{task}.csv",
        raw_dir / split / f"metadata_{task}_only.csv",
    ]
    metadata_csv = next((path for path in metadata_candidates if path.exists()), None)
    if metadata_csv is None:
        tried = ", ".join(str(path) for path in metadata_candidates)
        raise FileNotFoundError(
            f"Could not find VideoRLVR metadata for {split}/{task}. Tried: {tried}"
        )

    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {metadata_csv}")
    return task_dir, rows


def resolve_relative(task_dir: Path, raw_dir: Path, split: str, row_path: str) -> Path:
    relative = Path(row_path)
    candidates = (task_dir / relative, raw_dir / split / relative)
    return next((path for path in candidates if path.exists()), candidates[0])


def infer_input_image(task_dir: Path, video_path: Path) -> Path | None:
    images_dir = task_dir / "images"
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{video_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def convert_split_task(
    *,
    raw_dir: Path,
    output_root: Path,
    split: str,
    task: str,
    repo_id: str,
    revision: str,
    limit_per_task: int | None,
    expected_count: int | None,
) -> int:
    task_dir, rows = read_metadata(raw_dir, split, task)
    if limit_per_task is not None:
        rows = rows[:limit_per_task]
    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(
            f"Unexpected {split}/{task} size: found {len(rows)}, expected {expected_count}. "
            "Use --no-validate-counts only if this difference is intentional."
        )

    for subdir in OUTPUT_SUBDIRS:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    converted = 0
    for row_index, row in enumerate(rows):
        video_rel = row.get("video", "")
        prompt = row.get("prompt", "")
        if not video_rel or not prompt:
            raise RuntimeError(
                f"Bad row {row_index} in {split}/{task}: missing video or prompt"
            )

        source_video = resolve_relative(task_dir, raw_dir, split, video_rel)
        if not source_video.is_file():
            raise FileNotFoundError(f"Video listed in metadata does not exist: {source_video}")

        sample_id = source_video.stem
        sample_name = f"{task}__{sample_id}"
        destination_video = output_root / "video" / sample_name / "0.mp4"
        destination_caption = output_root / "caption" / sample_name / "0.json"
        destination_metadata = output_root / "metadata" / f"{sample_name}.json"
        destination_video.parent.mkdir(parents=True, exist_ok=True)
        destination_caption.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_video), str(destination_video))
        destination_caption.write_text(
            json.dumps({"caption": prompt}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        input_image_rel = row.get("input_image", "")
        input_image = (
            resolve_relative(task_dir, raw_dir, split, input_image_rel)
            if input_image_rel
            else infer_input_image(task_dir, source_video)
        )
        metadata = {
            "repo_id": repo_id,
            "revision": revision,
            "split": split,
            "task": task,
            "sample_id": sample_id,
            "source_video": video_rel,
            "source_input_image": input_image_rel,
            "input_image_found": bool(input_image and input_image.exists()),
            "prompt": prompt,
            "dataset_card_resolution": [480, 832],
            "dataset_card_frames": 81,
        }
        destination_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        converted += 1

    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "split": split,
        "task": task,
        "total_samples": converted,
        "layout": "LongLive video/caption/metadata",
        "video": {"dataset_card_resolution": [480, 832], "dataset_card_frames": 81},
    }
    task_manifest_path(output_root, task).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return converted


def write_root_manifest(
    output_root: Path,
    *,
    repo_id: str,
    revision: str,
    split: str,
    tasks: list[str],
    counts: dict[str, int],
) -> None:
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "split": split,
        "tasks": tasks,
        "counts": counts,
        "total_samples": sum(counts.values()),
        "layout": "LongLive video/caption/metadata",
        "video": {"dataset_card_resolution": [480, 832], "dataset_card_frames": 81},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    tasks = list(args.tasks)
    raw_root = resolve_path(args.raw_root)
    revision = resolve_revision(args.repo_id, args.revision, args.skip_download)
    print(f"Using {args.repo_id}@{revision}", flush=True)

    summary: dict[str, int] = {}
    for split in SPLITS:
        output_root = split_output_root(args, split)
        counts: dict[str, int] = {}
        expected_count = None
        if args.limit_per_task is None and not args.no_validate_counts:
            expected_count = EXPECTED_PER_TASK[split]

        for task in tasks:
            print(f"\n=== Processing VideoRLVR {split}/{task} ===", flush=True)
            if not args.overwrite and task_is_complete(
                output_root,
                repo_id=args.repo_id,
                revision=revision,
                split=split,
                task=task,
                expected_count=expected_count,
            ):
                manifest = json.loads(
                    task_manifest_path(output_root, task).read_text(encoding="utf-8")
                )
                count = int(manifest["total_samples"])
                print(f"Skipping complete task ({count} samples).", flush=True)
            else:
                clean_task_output(output_root, task)
                raw_dir = raw_task_dir(raw_root, split, task)
                if raw_dir.exists() and not args.skip_download:
                    shutil.rmtree(raw_dir)
                if not args.skip_download:
                    download_split_task(args.repo_id, revision, raw_dir, split, task)
                count = convert_split_task(
                    raw_dir=raw_dir,
                    output_root=output_root,
                    split=split,
                    task=task,
                    repo_id=args.repo_id,
                    revision=revision,
                    limit_per_task=args.limit_per_task,
                    expected_count=expected_count,
                )
                if raw_dir.exists() and not args.keep_raw:
                    shutil.rmtree(raw_dir)

            counts[task] = count
            summary[f"{split}/{task}"] = count

        write_root_manifest(
            output_root,
            repo_id=args.repo_id,
            revision=revision,
            split=split,
            tasks=tasks,
            counts=counts,
        )

    print("\nPrepared VideoRLVR datasets:")
    for key, count in summary.items():
        print(f"  {key}: {count}")
    print(f"Train root: {split_output_root(args, 'train')}")
    print(f"Test root:  {split_output_root(args, 'test')}")
    print(f"Pinned source revision: {revision}")


if __name__ == "__main__":
    main()
