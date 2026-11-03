import os
import zipfile
import shutil
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download
from src.utils.geometry import mot_to_yolo
import json
import re

from src.processing.frame_alignment import build_frame_mapping

def parse_annotations(gt_file_path: Path) -> dict[int, list[list[float]]]:
    """Parses ground truth MOT annotations file."""
    annotations = {}
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            f_id = int(parts[0])
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            annotations.setdefault(f_id, []).append(bbox)
    return annotations


def extract_frames_and_labels(
    video_path: Path,
    annotations: dict[int, list[list[float]]],
    video_base_name: str,
    pos_dir: Path,
    neg_dir: Path,
    review: dict,
) -> None:
    mapping = build_frame_mapping(review)

    if review.get("video_name") != video_base_name:
        raise ValueError("Review video_name does not match the video")

    if any(
        type(frame) is not int or not 1 <= frame <= len(mapping) or not boxes
        for frame, boxes in annotations.items()
    ):
        raise ValueError("Invalid MOT frame IDs or empty annotations")

    cap = cv2.VideoCapture(str(video_path))

    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened() or w <= 0 or h <= 0:
            raise ValueError(f"Could not open video: {video_path}")

        for frame_idx in range(1, review["size"] + 1):
            ret, frame = cap.read()

            if not ret:
                raise ValueError(f"Could not read source frame {frame_idx}")

            if frame_idx not in mapping:
                continue

            boxes = annotations.get(mapping[frame_idx], [])

            file_id = f"{video_base_name}_f{frame_idx:04d}"
            is_positive = bool(boxes)
            image_dir = pos_dir if is_positive else neg_dir
            image_path = image_dir / f"{file_id}.jpg"

            if not cv2.imwrite(str(image_path), frame):
                raise OSError(f"Could not save JPEG: {image_path}")

            if is_positive:
                label_path = pos_dir / f"{file_id}.txt"

                with label_path.open("w", encoding="utf-8") as output:
                    for box in boxes:
                        cx, cy, wn, hn = mot_to_yolo(box, w, h)
                        output.write(
                            f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n"
                        )

        ret, _ = cap.read()

        if ret:
            raise ValueError("Video contains more frames than review size")
    finally:
        cap.release()

def download_and_extract_assets(
    repo_id: str,
    hf_token: str,
    video_path_hf: str,
    zip_path_hf: str,
    tmp_dir: Path,
    video_base_name: str,
    revision: str,
) -> tuple[Path, Path, dict]:
    review_path_hf = zip_path_hf.removesuffix(".zip") + ".review.json"

    local_review = Path(hf_hub_download(
        repo_id=repo_id,
        filename=review_path_hf,
        local_dir=tmp_dir,
        repo_type="dataset",
        token=hf_token,
        revision=revision,
    ))

    with local_review.open(encoding="utf-8") as file:
        review = json.load(file)

    build_frame_mapping(review)

    if (
        review.get("video_name") != video_base_name
        or review.get("version") != zip_path_hf.split("/")[1]
    ):
        raise ValueError("Review identity does not match the annotation path")

    local_video = Path(hf_hub_download(
        repo_id=repo_id,
        filename=video_path_hf,
        local_dir=tmp_dir,
        repo_type="dataset",
        token=hf_token,
        revision=revision,
    ))

    local_zip = Path(hf_hub_download(
        repo_id=repo_id,
        filename=zip_path_hf,
        local_dir=tmp_dir,
        repo_type="dataset",
        token=hf_token,
        revision=revision,
    ))

    gt_temp_path = tmp_dir / f"{video_base_name}_gt.txt"

    with zipfile.ZipFile(local_zip) as archive:
        gt_files = [
            name
            for name in archive.namelist()
            if Path(name).name == "gt.txt"
        ]

        if len(gt_files) != 1:
            raise ValueError("Expected exactly one gt.txt in MOT archive")

        with archive.open(gt_files[0]) as source:
            with gt_temp_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    return local_video, gt_temp_path, review


def select_annotation_zips(
    repo_files: list[str],
    split: str,
    version: str,
    video_name: str | None = None,
) -> list[str]:
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError(f"Invalid annotation version: {version}")

    prefix = f"labels_curated/{version}/{split}/"

    archives = sorted(
        path
        for path in repo_files
        if path.startswith(prefix) and path.lower().endswith(".zip")
    )

    if video_name is not None:
        archives = [
            path
            for path in archives
            if Path(path).stem == video_name
        ]

        if len(archives) != 1:
            raise ValueError(
                f"Expected exactly one ZIP for "
                f"{split}/{video_name}, found {len(archives)}"
            )

    if not archives:
        raise ValueError(f"No curated annotations found for {version}/{split}")

    for path in archives:
        review_path = path.removesuffix(".zip") + ".review.json"

        if review_path not in repo_files:
            raise ValueError(f"Missing review metadata: {review_path}")

    if len({Path(path).stem for path in archives}) != len(archives):
        raise ValueError(f"Duplicate curated videos in {version}/{split}")

    return archives
