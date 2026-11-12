import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from src.processing.embedding_manifest import (
    file_sha256,
    save_embedding_selection as save_selection,
)
from src.processing.manifests import CATEGORY_DIRS, read_manifest
from src.processing.sampling import collect_frame_runs, select_block_indices


def build_temporal_manifest(rows, block_size):
    select_block_indices(0, block_size)
    selected = set()

    for category in CATEGORY_DIRS:
        runs = collect_frame_runs(rows, category)

        for run in runs.itertuples(index=False):
            indices = select_block_indices(
                len(run.filenames),
                block_size,
            )
            selected.update(
                (run.split, run.filenames[index])
                for index in indices
            )

    result = []

    for original in rows:
        row = dict(
            original,
            excluded_at=original.get("excluded_at", ""),
        )

        if (
            row["split"] in ("val", "test")
            and not row["excluded_at"]
            and (row["split"], row["filename"]) not in selected
        ):
            row["excluded_at"] = "temporal"

        result.append(row)

    return result


def verify_temporal_inputs(recorded, current):
    for field in (
        "source",
        "input_manifest_sha256",
        "input_run_sha256",
    ):
        if field not in recorded or recorded[field] != current[field]:
            raise ValueError(f"Reproduction input mismatch: {field}")

    digest = recorded.get("manifest_sha256")

    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Missing or invalid recorded manifest checksum")


def main():
    from src.config import MANIFESTS_DIR, PROJECT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--from-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    recorded = None

    if args.from_run is not None:
        if args.block_size is not None:
            parser.error(
                "--from-run cannot be combined with --block-size"
            )

        if args.output_dir is None:
            parser.error("--from-run requires --output-dir")

        recorded = json.loads(
            args.from_run.read_text(encoding="utf-8")
        )
        block_size = recorded["block_size"]
    else:
        if args.block_size is None:
            parser.error("Provide --block-size or --from-run")

        block_size = args.block_size

    select_block_indices(0, block_size)

    source = json.loads(
        (MANIFESTS_DIR / "sources" / "hf.json").read_text(
            encoding="utf-8"
        )
    )
    version = source["annotation_version"]

    input_dir = MANIFESTS_DIR / "embeddings" / version / "within_video"
    input_manifest = input_dir / "main.csv"
    input_run_path = input_dir / "run.json"
    input_run = json.loads(
        input_run_path.read_text(encoding="utf-8")
    )
    input_checksum = file_sha256(input_manifest)

    if input_run["source"] != source:
        raise ValueError("Input selection source mismatch")

    if input_checksum != input_run["manifest_sha256"]:
        raise ValueError("Input selection manifest checksum mismatch")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else MANIFESTS_DIR / "temporal" / version
    )

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Selection already exists: {output_dir}")

    metadata = {
        "source": source,
        "block_size": block_size,
        "input_manifest_sha256": input_checksum,
        "input_run_sha256": file_sha256(input_run_path),
        "code_revision": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "numpy_version": np.__version__,
    }

    if recorded is not None:
        verify_temporal_inputs(recorded, metadata)

    rows = build_temporal_manifest(
        read_manifest(input_manifest),
        block_size,
    )
    expected_checksum = (
        recorded["manifest_sha256"]
        if recorded is not None
        else None
    )

    save_selection(
        rows,
        metadata,
        output_dir,
        expected_manifest_sha256=expected_checksum,
    )

    print(f"Manifest: {output_dir / 'main.csv'}")

    counts = Counter(
        (row["split"], row["category"])
        for row in rows
        if not row["excluded_at"]
    )

    for (split, category), count in sorted(counts.items()):
        print(f"{split}/{category}: {count}")

    if recorded is not None:
        print("Reproduced manifest checksum matches the recorded result")


if __name__ == "__main__":
    main()