#!/usr/bin/env python3
"""Download and convert the VBVR 10-task training subset.

The source is ``Video-Reason/VBVR-Dataset`` on Hugging Face. The output uses
the layout consumed by ``MultiVideoConcatDataset``::

    data/vbvr-datasets-10tasks/
      video/<task>__<sample>/0.mp4
      caption/<task>__<sample>/0.json
      metadata/<task>__<sample>.json

Videos are hardlinked from the extraction cache when possible and copied when
the cache and output directory are on different filesystems.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


DEFAULT_REPO_ID = "Video-Reason/VBVR-Dataset"
DEFAULT_TASKS = (
    "G-194",
    "G-25",
    "G-3",
    "G-5",
    "O-23",
    "O-31",
    "O-36",
    "O-44",
    "O-47",
    "O-75",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset branch, tag, or commit. The resolved commit SHA is recorded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "data" / "vbvr-datasets-10tasks",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root() / "data" / ".cache" / "vbvr-raw",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, check=True)


def resolve_revision(repo_id: str, revision: str, skip_download: bool) -> str:
    if skip_download:
        return revision
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before downloading VBVR.") from exc
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve {repo_id}@{revision} to a commit SHA")
    return str(info.sha)


def task_prefix(name: str) -> str:
    return name.split("_", 1)[0]


def download_task_archives(
    repo_id: str,
    revision: str,
    raw_dir: Path,
    tasks: list[str],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "hf",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(raw_dir),
    ]
    for task in tasks:
        command.extend(["--include", f"tars/{task}_*.tar"])
    run(command)


def safe_extract(archive: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (output_dir / member.name).resolve()
            if target != output_root and output_root not in target.parents:
                raise RuntimeError(f"Unsafe path in archive {archive}: {member.name}")
        handle.extractall(output_dir)


def extract_task_archives(raw_dir: Path, tasks: list[str]) -> None:
    selected = set(tasks)
    already_extracted = {
        task_prefix(path.name) for path in find_task_dirs(raw_dir, tasks)
    }
    archives = [
        path
        for path in sorted((raw_dir / "tars").glob("*.tar"))
        if task_prefix(path.stem) in selected
    ]
    found = {task_prefix(path.stem) for path in archives}
    missing = [
        task for task in tasks if task not in found and task not in already_extracted
    ]
    if missing:
        raise FileNotFoundError(f"Missing VBVR task archives: {', '.join(missing)}")

    for archive in archives:
        existing = find_task_dirs(raw_dir, [task_prefix(archive.stem)])
        if existing:
            continue
        print(f"Extracting {archive}", flush=True)
        tar_binary = shutil.which("tar")
        if tar_binary:
            run([tar_binary, "-xf", str(archive), "-C", str(raw_dir)])
        else:
            safe_extract(archive, raw_dir)


def find_task_dirs(raw_dir: Path, tasks: list[str]) -> list[Path]:
    selected = set(tasks)
    return sorted(
        path
        for path in raw_dir.rglob("*_data-generator")
        if path.is_dir() and task_prefix(path.name) in selected
    )


def find_samples(task_dir: Path) -> list[Path]:
    samples = {
        path.parent
        for pattern in ("ground_truth.mp4", "ground_truth_h264.mp4")
        for path in task_dir.rglob(pattern)
    }
    return sorted(samples, key=lambda path: path.relative_to(task_dir).as_posix())


def place_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def task_manifest_path(output_dir: Path, task: str) -> Path:
    return output_dir / "manifests" / f"{task}.json"


def task_is_complete(
    output_dir: Path,
    task: str,
    repo_id: str,
    revision: str,
) -> bool:
    manifest_path = task_manifest_path(output_dir, task)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    count = int(manifest.get("total_samples", 0))
    if not all(
        (
            manifest.get("repo_id") == repo_id,
            manifest.get("revision") == revision,
            manifest.get("task_prefix") == task,
            count > 0,
        )
    ):
        return False
    video_count = len(list((output_dir / "video").glob(f"{task}_*/0.mp4")))
    caption_count = len(list((output_dir / "caption").glob(f"{task}_*/0.json")))
    metadata_count = len(list((output_dir / "metadata").glob(f"{task}_*.json")))
    return video_count == caption_count == metadata_count == count


def clean_task(output_dir: Path, task: str) -> None:
    for root_name in ("video", "caption"):
        for path in (output_dir / root_name).glob(f"{task}_*"):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    for path in (output_dir / "metadata").glob(f"{task}_*.json"):
        path.unlink()
    manifest = task_manifest_path(output_dir, task)
    if manifest.exists():
        manifest.unlink()


def convert_task(
    task_dir: Path,
    output_dir: Path,
    repo_id: str,
    revision: str,
    link_mode: str,
) -> tuple[str, int]:
    prefix = task_prefix(task_dir.name)
    samples = find_samples(task_dir)
    if not samples:
        raise RuntimeError(f"No ground-truth videos found under {task_dir}")
    sample_names = [sample.name for sample in samples]
    if len(sample_names) != len(set(sample_names)):
        raise RuntimeError(f"Duplicate sample names found under {task_dir}")

    for sample_dir in samples:
        source_video = sample_dir / "ground_truth_h264.mp4"
        if not source_video.is_file():
            source_video = sample_dir / "ground_truth.mp4"
        prompt_path = sample_dir / "prompt.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing prompt: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Empty prompt: {prompt_path}")

        sample_name = f"{task_dir.name}__{sample_dir.name}"
        destination_video = output_dir / "video" / sample_name / "0.mp4"
        destination_caption = output_dir / "caption" / sample_name / "0.json"
        destination_metadata = output_dir / "metadata" / f"{sample_name}.json"
        place_file(source_video, destination_video, link_mode)
        destination_caption.parent.mkdir(parents=True, exist_ok=True)
        destination_caption.write_text(
            json.dumps({"caption": prompt}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        destination_metadata.parent.mkdir(parents=True, exist_ok=True)
        destination_metadata.write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "revision": revision,
                    "task": task_dir.name,
                    "task_prefix": prefix,
                    "sample_id": sample_dir.name,
                    "source_video": source_video.name,
                    "prompt": prompt,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "task": task_dir.name,
        "task_prefix": prefix,
        "total_samples": len(samples),
        "layout": "LongLive video/caption/metadata",
    }
    manifest_path = task_manifest_path(output_dir, prefix)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return task_dir.name, len(samples)


def main() -> None:
    args = parse_args()
    tasks = list(dict.fromkeys(args.tasks))
    revision = resolve_revision(args.repo_id, args.revision, args.skip_download)
    print(f"Using {args.repo_id}@{revision}", flush=True)

    if not args.skip_download:
        download_task_archives(args.repo_id, revision, args.raw_dir, tasks)
    extract_task_archives(args.raw_dir, tasks)

    task_dirs = find_task_dirs(args.raw_dir, tasks)
    directories_by_prefix: dict[str, list[Path]] = {task: [] for task in tasks}
    for task_dir in task_dirs:
        directories_by_prefix[task_prefix(task_dir.name)].append(task_dir)

    counts: dict[str, int] = {}
    for task in tasks:
        matches = directories_by_prefix[task]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one extracted directory for {task}, found {len(matches)}: {matches}"
            )
        if not args.overwrite and task_is_complete(
            args.output_dir, task, args.repo_id, revision
        ):
            manifest = json.loads(
                task_manifest_path(args.output_dir, task).read_text(encoding="utf-8")
            )
            counts[str(manifest["task"])] = int(manifest["total_samples"])
            print(f"Skipping complete task {task}.", flush=True)
            continue

        clean_task(args.output_dir, task)
        task_name, count = convert_task(
            matches[0], args.output_dir, args.repo_id, revision, args.link_mode
        )
        counts[task_name] = count
        print(f"Prepared {task_name}: {count} samples", flush=True)

    root_manifest = {
        "repo_id": args.repo_id,
        "revision": revision,
        "task_prefixes": tasks,
        "counts": counts,
        "total_samples": sum(counts.values()),
        "layout": "LongLive video/caption/metadata",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done: {sum(counts.values())} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
