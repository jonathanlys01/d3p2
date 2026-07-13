"""Create the derivative model repository without transferring weights locally."""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi


SOURCE_REPO = "GSAI-ML/LLaDA-8B-Instruct"
TARGET_REPO = "jonathanlys01/LLaDA-8B-Instruct-D5P4"
CUSTOM_FILES = (
    "README.md",
    "config.py",
    "dpp.py",
    "inference.py",
    "sampler.py",
    "create_repo.py",
    "requirements.txt",
)
SHA_PLACEHOLDER = "{{UPSTREAM_COMMIT_SHA}}"
SHARD_PATTERN = re.compile(r"model-\d{5}-of-00006\.safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the custom repository files.",
    )
    return parser.parse_args()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _file_metadata(api: HfApi, repo_id: str, revision: str | None = None) -> dict[str, Any]:
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    return {sibling.rfilename: sibling for sibling in (info.siblings or [])}


def verify_remote_copy(api: HfApi, source_revision: str) -> None:
    """Compare weight presence, sizes, and LFS hashes without downloading a file."""
    source = _file_metadata(api, SOURCE_REPO, source_revision)
    destination = _file_metadata(api, TARGET_REPO)
    source_shards = sorted(path for path in source if SHARD_PATTERN.fullmatch(path))
    if len(source_shards) != 6:
        raise RuntimeError(f"Expected six upstream Safetensors shards, found {len(source_shards)}")
    required = [*source_shards, "model.safetensors.index.json"]
    missing = [path for path in required if path not in destination]
    if missing:
        raise RuntimeError(f"Destination is missing required files: {missing}")

    for path in source_shards:
        source_file = source[path]
        destination_file = destination[path]
        source_lfs = _field(source_file, "lfs")
        destination_lfs = _field(destination_file, "lfs")
        source_sha = _field(source_lfs, "sha256")
        destination_sha = _field(destination_lfs, "sha256")
        if _field(source_file, "size") != _field(destination_file, "size"):
            raise RuntimeError(f"Size mismatch for {path}")
        if not source_sha or source_sha != destination_sha:
            raise RuntimeError(f"LFS SHA-256 mismatch for {path}")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    missing = [name for name in CUSTOM_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing custom files in {source_dir}: {missing}")

    api = HfApi()
    source_info = api.model_info(SOURCE_REPO, revision="main")
    source_revision = source_info.sha
    if source_revision is None:
        raise RuntimeError("Could not resolve the upstream commit SHA.")

    files = api.list_repo_files(
        repo_id=SOURCE_REPO,
        repo_type="model",
        revision=source_revision,
    )
    api.create_repo(repo_id=TARGET_REPO, repo_type="model", exist_ok=False)
    copy_commit = api.create_commit(
        repo_id=TARGET_REPO,
        repo_type="model",
        operations=[
            CommitOperationCopy(
                src_repo_id=SOURCE_REPO,
                src_repo_type="model",
                src_revision=source_revision,
                src_path_in_repo=path,
                path_in_repo=path,
            )
            for path in files
        ],
        commit_message=f"Copy {SOURCE_REPO}@{source_revision}",
    )

    buffers: list[io.BytesIO] = []
    add_operations: list[CommitOperationAdd] = []
    for name in CUSTOM_FILES:
        contents = (source_dir / name).read_bytes()
        if name == "README.md":
            text = contents.decode("utf-8")
            if SHA_PLACEHOLDER not in text:
                raise RuntimeError(f"README.md must contain {SHA_PLACEHOLDER}")
            contents = text.replace(SHA_PLACEHOLDER, source_revision).encode("utf-8")
        buffer = io.BytesIO(contents)
        buffers.append(buffer)
        add_operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=buffer))

    custom_commit = api.create_commit(
        repo_id=TARGET_REPO,
        repo_type="model",
        operations=add_operations,
        commit_message="Add custom LLaDA sampler and inference script",
        parent_commit=copy_commit.oid,
    )
    verify_remote_copy(api, source_revision)
    print(f"Source revision: {source_revision}")
    print(f"Destination commit: {custom_commit.commit_url}")


if __name__ == "__main__":
    main()
