import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.processing.clustering import k_center_greedy_threshold
from src.processing.manifests import CATEGORY_DIRS, read_manifest, write_manifest


def build_embedding_manifest(rows, embeddings_by_category, thresholds):
    if set(thresholds) != set(CATEGORY_DIRS):
        raise ValueError("Provide a threshold for each category")

    if any(not np.isfinite(value) or value < 0 for value in thresholds.values()):
        raise ValueError("Thresholds must be finite and nonnegative")

    if set(embeddings_by_category) != set(CATEGORY_DIRS):
        raise ValueError("Provide embeddings for each category")

    result = [
        dict(row, excluded_at=row.get("excluded_at", ""))
        for row in rows
    ]

    for category in CATEGORY_DIRS:
        category_rows = [
            row for row in result
            if row["split"] == "train" and row["category"] == category
        ]
        active = [row for row in category_rows if not row["excluded_at"]]
        embeddings = embeddings_by_category[category]
        expected = {row["filename"] for row in active}
        allowed = {row["filename"] for row in category_rows}

        if not expected <= set(embeddings) <= allowed:
            raise ValueError(f"Embedding filenames mismatch for {category}")

        groups = defaultdict(list)

        for row in active:
            name = row["filename"]
            video_id, separator, number = Path(name).stem.rpartition("_f")

            if (
                not video_id
                or not separator
                or not number.isdecimal()
                or int(number) < 1
            ):
                raise ValueError(f"Invalid frame filename: {name}")

            groups[video_id].append((int(number), row))

        for video_id in sorted(groups):
            frames = sorted(groups[video_id], key=lambda item: item[0])

            if len({number for number, _ in frames}) != len(frames):
                raise ValueError(f"Duplicate frame number in {video_id}")

            matrix = np.stack([
                embeddings[row["filename"]]
                for _, row in frames
            ])

            if (
                matrix.ndim != 2
                or not matrix.shape[1]
                or not np.issubdtype(matrix.dtype, np.floating)
                or not np.isfinite(matrix).all()
                or not np.allclose(
                    np.linalg.norm(matrix, axis=1),
                    1,
                    rtol=1e-5,
                    atol=1e-6,
                )
            ):
                raise ValueError(f"Invalid normalized embeddings for {video_id}")

            selected = set(
                k_center_greedy_threshold(matrix, thresholds[category])
            )

            for index, (_, row) in enumerate(frames):
                if index not in selected:
                    row["excluded_at"] = "embeddings"

    return result


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_embedding_selection(
    rows,
    metadata,
    output_dir,
    *,
    expected_manifest_sha256=None,
):
    output_dir = Path(output_dir)

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Selection already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        dir=output_dir.parent,
        prefix=".embedding-selection-",
    ) as temporary:
        staging = Path(temporary) / "selection"
        write_manifest(rows, staging / "main.csv")
        manifest_sha256 = file_sha256(staging / "main.csv")

        if (
            expected_manifest_sha256 is not None
            and manifest_sha256 != expected_manifest_sha256
        ):
            raise ValueError(
                "Reproduced manifest differs from the recorded result"
            )

        record = dict(metadata, manifest_sha256=manifest_sha256)
        (staging / "run.json").write_text(
            json.dumps(
                record,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        staging.rename(output_dir)



def verify_run_inputs(recorded, current):
    fields = (
        "source",
        "baseline_sha256",
        "embedding_files_sha256",
    )

    for field in fields:
        if field not in recorded or recorded[field] != current[field]:
            raise ValueError(f"Reproduction input mismatch: {field}")

    digest = recorded.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Missing or invalid recorded manifest checksum")



def main():
    import subprocess
    import torch

    from src.config import (
        EMBEDDINGS_DIR,
        FRAME_BASELINE_DIR,
        MANIFESTS_DIR,
        PROJECT_ROOT,
    )
    from src.processing.embeddings import validate_embeddings

    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-theta", type=float)
    parser.add_argument("--negative-theta", type=float)
    parser.add_argument("--from-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    recorded = None

    if args.from_run is not None:
        if args.positive_theta is not None or args.negative_theta is not None:
            parser.error(
                "--from-run cannot be combined with threshold arguments"
            )

        if args.output_dir is None:
            parser.error("--from-run requires --output-dir")

        recorded = json.loads(
            args.from_run.read_text(encoding="utf-8")
        )
        thresholds = recorded["thresholds"]
    else:
        if args.positive_theta is None or args.negative_theta is None:
            parser.error(
                "Provide both --positive-theta and --negative-theta"
            )

        thresholds = {
            "positive": args.positive_theta,
            "negative": args.negative_theta,
        }

    source_path = MANIFESTS_DIR / "sources" / "hf.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    version = source["annotation_version"]
    baseline = MANIFESTS_DIR / "baseline" / version / "main.csv"

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else MANIFESTS_DIR / "embeddings" / version / "within_video"
    )

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Selection already exists: {output_dir}")

    if recorded is not None and recorded.get("source") != source:
        raise ValueError("Reproduction input mismatch: source")

    files = validate_embeddings(
        FRAME_BASELINE_DIR / version,
        source,
        manifest_path=baseline,
    )

    metadata = {
        "source": source,
        "thresholds": thresholds,
        "baseline_sha256": file_sha256(baseline),
        "embedding_files_sha256": {
            path.name: file_sha256(path)
            for path in files
        },
        "code_revision": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "numpy_version": np.__version__,
    }

    if recorded is not None:
        verify_run_inputs(recorded, metadata)

    embeddings = {
        category: torch.load(
            EMBEDDINGS_DIR / version / f"train_{category}s_embeddings.pt",
            map_location="cpu",
            weights_only=False,
        )
        for category in CATEGORY_DIRS
    }

    rows = build_embedding_manifest(
        read_manifest(baseline),
        embeddings,
        thresholds,
    )

    expected_checksum = (
        recorded["manifest_sha256"]
        if recorded is not None
        else None
    )

    save_embedding_selection(
        rows,
        metadata,
        output_dir,
        expected_manifest_sha256=expected_checksum,
    )

    print(f"Manifest: {output_dir / 'main.csv'}")

    if recorded is not None:
        print("Reproduced manifest checksum matches the recorded result")

    counts = Counter(
        (row["split"], row["category"])
        for row in rows
        if not row["excluded_at"]
    )

    for (split, category), count in sorted(counts.items()):
        print(f"{split}/{category}: {count}")


if __name__ == "__main__":
    main()