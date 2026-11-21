from io import BytesIO
import json
import logging
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np

from src.config import MANIFESTS_DIR
from src.processing.frame_alignment import build_frame_mapping
from src.processing.manifests import load_baseline_video_sequences
from src.processing.video import get_video_metadata

log = logging.getLogger(__name__)


def _validate_sequence_metadata(
    seq_name: str,
    width: int | float,
    height: int | float,
    fps: float,
    length: int | float,
) -> tuple[int, int, float, int]:
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or not math.isfinite(fps)
        or not math.isfinite(length)
        or int(width) <= 0
        or int(height) <= 0
        or float(fps) <= 0.0
        or int(length) <= 0
    ):
        raise ValueError(
            f"Invalid sequence metadata for {seq_name}: w={width}, h={height}, fps={fps}, length={length}"
        )
    return int(width), int(height), float(fps), int(length)


def prepare_trackeval_sequence(
    video_path: Path,
    zip_path: Path,
    seq_gt_dir: Path,
    seqinfo_path: Path,
    *,
    review: dict | None = None,
    sequence_metadata: tuple[int, int, float, int] | None = None,
) -> None:
    """Extracts curated MOT ground truth, validates metadata, aligns retained length, and writes TrackEval files."""
    video_path = Path(video_path)
    zip_path = Path(zip_path)
    seq_name = video_path.stem

    raw_meta = (
        get_video_metadata(str(video_path))
        if sequence_metadata is None
        else sequence_metadata
    )
    width, height, fps, length = _validate_sequence_metadata(seq_name, *raw_meta)

    seq_length = length
    mapping: dict[int, int] | None = None
    if review is not None:
        if review.get("video_name") != seq_name:
            raise ValueError(
                f"Review video_name {review.get('video_name')!r} does not match sequence {seq_name!r}"
            )
        mapping = build_frame_mapping(review)
        expected_len = len(mapping)
        expected_raw = review["size"] if sequence_metadata is None else expected_len
        if length != expected_raw:
            raise ValueError(
                f"Frame count mismatch for {seq_name}: got {length}, expected {expected_raw}"
            )
        seq_length = expected_len

    seqinfo_content = (
        f"[Sequence]\nname={seq_name}\nimDir=img1\n"
        f"frameRate={int(round(fps))}\nseqLength={seq_length}\n"
        f"imWidth={width}\nimHeight={height}\nimExt=.jpg\n"
    )

    with zipfile.ZipFile(zip_path) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and Path(entry.filename).name == "gt.txt"
        ]
        if len(entries) != 1:
            raise ValueError(
                f"{zip_path}: expected exactly one gt.txt, found {len(entries)}"
            )
        contents = archive.read(entries[0])

    if contents.strip():
        data = np.loadtxt(BytesIO(contents), delimiter=",", ndmin=2)
        if data.shape[1] != 9:
            raise ValueError(f"{zip_path}: expected 9 columns in gt.txt")
        frame_ids = data[:, 0]
        if (
            not np.all(np.isfinite(frame_ids))
            or np.any(frame_ids != np.floor(frame_ids))
            or np.any(frame_ids < 1)
            or np.any(frame_ids > seq_length)
        ):
            raise ValueError(
                f"{zip_path}: gt.txt frame IDs must be integers in [1, {seq_length}]"
            )
        data[:, 7] = 1
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        target = seq_gt_dir / "gt.txt"
        np.savetxt(
            target,
            data,
            fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f",
        )
    else:
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        (seq_gt_dir / "gt.txt").write_bytes(b"")

    seqinfo_path.parent.mkdir(parents=True, exist_ok=True)
    seqinfo_path.write_text(seqinfo_content, encoding="utf-8")


def _collect_curated_split_archives(
    curated_split_dir: Path,
) -> dict[str, tuple[Path, dict]]:
    archives: dict[str, tuple[Path, dict]] = {}
    for zip_path in sorted(curated_split_dir.rglob("*.zip")):
        seq_name = zip_path.stem
        if seq_name in archives:
            raise ValueError(f"Duplicate curated archive for sequence {seq_name!r} in {curated_split_dir}")
        review_path = Path(str(zip_path).removesuffix(".zip") + ".review.json")
        if not review_path.is_file():
            raise FileNotFoundError(f"Missing curated review JSON: {review_path}")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        build_frame_mapping(review)
        archives[seq_name] = (zip_path, review)
    return archives


def _inspect_baseline_sequence_metadata(
    seq_name: str,
    seq_frames: list[dict[str, object]],
    review: dict,
    default_fps: float,
) -> tuple[int, int, float, int]:
    mapping = build_frame_mapping(review)
    actual_source_frames = [int(frame["source_frame"]) for frame in seq_frames]
    expected_source_frames = list(mapping)
    if actual_source_frames != expected_source_frames:
        raise ValueError(
            f"Baseline frames do not match curated review mapping for {seq_name}"
        )
    first_img = cv2.imread(str(seq_frames[0]["image_path"]))
    if first_img is None:
        raise ValueError(f"Unreadable baseline image: {seq_frames[0]['image_path']}")
    height, width = first_img.shape[:2]
    return width, height, default_fps, len(seq_frames)


def compile_kaggle_mot_dataset(
    local_hf_dir: Path,
    mot_dataset_dir: Path,
    seqmaps_dir: Path,
    video_extensions: tuple[str, ...],
    *,
    annotation_version: str = "v1",
    splits: tuple[str, ...] = ("test", "val"),
    baseline_dataset_root: Path | None = None,
    manifest_path: Path = MANIFESTS_DIR / "baseline" / "v1" / "main.csv",
    source_path: Path | None = None,
    default_fps: float = 30.0,
) -> dict[str, list[str]]:
    """Compiles TrackEval GT strictly from curated annotations and reviews with optional baseline frames."""
    local_hf_dir = Path(local_hf_dir)
    mot_dataset_dir = Path(mot_dataset_dir)
    seqmaps_dir = Path(seqmaps_dir)
    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    ext_set = {ext.lower() for ext in video_extensions}

    curated_version_dir = local_hf_dir / "labels_curated" / annotation_version
    if not curated_version_dir.is_dir():
        raise FileNotFoundError(f"Curated annotations directory not found: {curated_version_dir}")

    compiled: dict[str, list[str]] = {}
    for split in splits:
        curated_split_dir = curated_version_dir / split
        if not curated_split_dir.is_dir():
            log.warning("Curated split directory not found, skipping: %s", curated_split_dir)
            continue

        curated = _collect_curated_split_archives(curated_split_dir)
        trackeval_split_name = f"drone-{split}"
        gt_dir = mot_dataset_dir / trackeval_split_name
        gt_dir.mkdir(parents=True, exist_ok=True)

        video_by_stem: dict[str, Path] = {}
        split_video_dir = local_hf_dir / split
        if split_video_dir.is_dir():
            for path in sorted(split_video_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in ext_set:
                    if path.stem in video_by_stem:
                        raise ValueError(f"Duplicate video stem {path.stem!r} in {split_video_dir}")
                    video_by_stem[path.stem] = path

        baseline_seqs = (
            load_baseline_video_sequences(
                Path(baseline_dataset_root),
                split=split,
                manifest_path=manifest_path,
                source_path=source_path,
            )
            if baseline_dataset_root is not None
            else {}
        )

        matched_seqs = []
        for seq_name in sorted(curated):
            zip_path, review = curated[seq_name]
            video_path = video_by_stem.get(seq_name)
            seq_meta: tuple[int, int, float, int] | None = None

            if video_path is None:
                seq_frames = baseline_seqs.get(seq_name)
                if not seq_frames:
                    log.warning("No video or baseline frames found for curated sequence %s (%s)", seq_name, split)
                    continue
                video_path = Path(f"{seq_name}.mp4")
                seq_meta = _inspect_baseline_sequence_metadata(
                    seq_name,
                    seq_frames,
                    review,
                    default_fps=default_fps,
                )

            seq_gt_dir = gt_dir / seq_name / "gt"
            seqinfo_path = gt_dir / seq_name / "seqinfo.ini"
            prepare_trackeval_sequence(
                video_path,
                zip_path,
                seq_gt_dir,
                seqinfo_path,
                review=review,
                sequence_metadata=seq_meta,
            )
            matched_seqs.append(seq_name)

        (seqmaps_dir / f"{trackeval_split_name}.txt").write_text(
            "name\n" + "\n".join(matched_seqs),
            encoding="utf-8",
        )
        compiled[split] = matched_seqs
        log.info("Compiled %d curated TrackEval sequences for %s", len(matched_seqs), trackeval_split_name)

    if not compiled:
        raise FileNotFoundError(f"No curated splits found under {curated_version_dir}")

    return compiled