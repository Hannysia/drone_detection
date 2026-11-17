from collections import Counter
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.processing.dataset import extract_metadata_from_filename, parse_frame_filename_logic
from src.processing.video import get_video_metadata

log = logging.getLogger(__name__)


def analyze_video_formats(root_path):
    if not os.path.exists(root_path):
        print("Error: The specified path does not exist. Please check the drive letter and folder name.")
        return

    extension_counter = Counter()
    total_files = 0
    total_folders = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        total_folders += 1
        for file in filenames:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()

            if ext == "":
                ext = "no_extension"

            extension_counter[ext] += 1

    return extension_counter, total_folders, total_files


def get_dataset_durations_metadata(source_dir: Path, video_extensions: list[str]) -> list[dict]:
    """Gathers filename, absolute path, and duration for all valid video files."""
    durations_data = []

    video_files = [
        f for f in source_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    for video_path in video_files:
        try:
            _, _, fps, length = get_video_metadata(str(video_path))
        except (OSError, cv2.error, ValueError, RuntimeError) as exc:
            log.error("Failed to read video metadata for %s: %s", video_path, exc)
            raise RuntimeError(f"Failed to read video metadata for {video_path}") from exc

        if fps <= 0 or length <= 0:
            raise ValueError(
                f"Invalid video metadata for {video_path}: fps={fps}, length={length}"
            )

        duration = length / fps
        durations_data.append({
            "filename": video_path.name,
            "path": str(video_path),
            "duration_sec": duration,
        })

    return durations_data


def get_splits_metadata_df(split_dir: Path, splits: list[str]) -> pd.DataFrame:
    """Scans all split folders, parses video filenames, and returns a single Pandas DataFrame."""
    all_data = []

    for split in splits:
        folder_path = split_dir / split
        if not folder_path.exists():
            continue

        for video_path in folder_path.iterdir():
            meta = extract_metadata_from_filename(video_path.name)
            if meta:
                shape, camera, bg = meta
                all_data.append({
                    "Class": f"{shape}_{camera}_{bg}",
                    "Morphology": shape,
                    "Camera": camera,
                    "Background": bg,
                    "Split": split.upper(),
                })

    return pd.DataFrame(all_data)


def gather_dataset_geometry(
    pos_dir: Path,
    *,
    selected_frames: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scans positive images and labels, calculates geometry dimensions and scales.
    Returns:
        - df_images: DataFrame with original widths, heights, and aspect ratios.
        - df_bboxes: DataFrame with pixel-level width, height, area, and relative scales.
    """
    img_widths = []
    img_heights = []
    img_aspect_ratios = []

    bbox_pixel_widths = []
    bbox_pixel_heights = []
    bbox_pixel_areas = []
    bbox_relative_areas = []
    bbox_aspect_ratios = []

    image_paths = sorted(list(pos_dir.glob("*.jpg")))

    for img_path in image_paths:
        if (
            selected_frames is not None
            and (pos_dir.parent.name, img_path.name) not in selected_frames
        ):
            continue

        lbl_path = img_path.with_suffix(".txt")

        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as e:
            log.warning(f"Failed to read image header {img_path.name}: {e}")
            continue

        img_widths.append(w)
        img_heights.append(h)
        img_aspect_ratios.append(w / h)

        if lbl_path.exists():
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    _, _, _, nw, nh = map(float, parts)

                    pixel_w = nw * w
                    pixel_h = nh * h
                    pixel_area = pixel_w * pixel_h
                    relative_area = (pixel_area / (w * h)) * 100

                    bbox_pixel_widths.append(pixel_w)
                    bbox_pixel_heights.append(pixel_h)
                    bbox_pixel_areas.append(pixel_area)
                    bbox_relative_areas.append(relative_area)
                    bbox_aspect_ratios.append(pixel_w / (pixel_h + 1e-8))
            except Exception as e:
                log.warning(f"Failed to parse label {lbl_path.name}: {e}")

    df_images = pd.DataFrame({
        "width": img_widths,
        "height": img_heights,
        "aspect_ratio": img_aspect_ratios,
    })

    df_bboxes = pd.DataFrame({
        "pixel_width": bbox_pixel_widths,
        "pixel_height": bbox_pixel_heights,
        "pixel_area": bbox_pixel_areas,
        "relative_area_pct": bbox_relative_areas,
        "aspect_ratio": bbox_aspect_ratios,
    })

    return df_images, df_bboxes


def gather_spatial_coordinates(
    extracted_data_dir: Path,
    splits: list[str],
    *,
    selected_frames: set[tuple[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Scans all split positive labels and extracts normalized bounding box coordinates.
    Returns:
        - x_centers: Numpy array of normalized X coordinates clipped to [0.0, 1.0].
        - y_centers: Numpy array of normalized Y coordinates clipped to [0.0, 1.0].
        - total_boxes: Integer representing total successfully parsed bounding boxes.
    """
    x_centers = []
    y_centers = []
    total_boxes = 0

    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            continue

        for path in pos_dir.glob("*.txt"):
            if (
                selected_frames is not None
                and (
                    split,
                    path.with_suffix(".jpg").name,
                ) not in selected_frames
            ):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            x_c = float(parts[1])
                            y_c = float(parts[2])

                            x_centers.append(np.clip(x_c, 0.0, 1.0))
                            y_centers.append(np.clip(y_c, 0.0, 1.0))
                            total_boxes += 1
            except Exception as e:
                log.warning(f"Failed to parse spatial labels for file {path.name}: {e}")

    return np.array(x_centers), np.array(y_centers), total_boxes


def gather_dataset_frame_metadata_df(
    extracted_data_dir: Path,
    splits: list[str],
    *,
    selected_frames: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Scans positive and negative image directories across splits,
    extracts frame-level metadata, and returns a compiled Pandas DataFrame.
    """
    data_records = []

    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        neg_dir = extracted_data_dir / split / "RAW_NEGATIVES"

        if pos_dir.exists():
            for img_path in pos_dir.glob("*.jpg"):
                if (
                    selected_frames is not None
                    and (split, img_path.name) not in selected_frames
                ):
                    continue
                meta = parse_frame_filename_logic(img_path.name)
                data_records.append({
                    "filename": img_path.name,
                    "split": split,
                    "is_positive": True,
                    "shape": meta["shape"],
                    "camera": meta["camera"],
                    "background": meta["background"],
                    "combined_class": f"{meta['shape']}_{meta['camera']}_{meta['background']}",
                })

        if neg_dir.exists():
            for img_path in neg_dir.glob("*.jpg"):
                if (
                    selected_frames is not None
                    and (split, img_path.name) not in selected_frames
                ):
                    continue
                meta = parse_frame_filename_logic(img_path.name)
                data_records.append({
                    "filename": img_path.name,
                    "split": split,
                    "is_positive": False,
                    "shape": "no_drone",
                    "camera": meta["camera"],
                    "background": meta["background"],
                    "combined_class": f"no_drone_{meta['camera']}_{meta['background']}",
                })

    return pd.DataFrame(data_records)


def gather_all_splits_geometry(
    extracted_data_dir: Path,
    splits: list[str],
    *,
    selected_frames: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Scans and gathers geometry statistics (width, height, area, aspect ratio)
    for all bounding boxes across multiple splits. Uses fast PIL header scanner.
    """
    widths_px = []
    heights_px = []
    areas_px = []
    aspect_ratios = []

    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            continue

        image_paths = sorted(list(pos_dir.glob("*.jpg")))
        for img_path in image_paths:

            if (
                selected_frames is not None
                and (pos_dir.parent.name, img_path.name) not in selected_frames
            ):
                continue

            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                continue

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except (OSError, ValueError) as exc:
                log.error("Failed to read image header %s: %s", img_path, exc)
                raise ValueError(f"Unreadable image header: {img_path}") from exc

            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid image dimensions for {img_path}: {w}x{h}")

            with open(lbl_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        raise ValueError(
                            f"{lbl_path}:{line_no}: expected 5 YOLO fields"
                        )
                    try:
                        _, _, _, nw, nh = map(float, parts)
                    except ValueError as exc:
                        raise ValueError(
                            f"{lbl_path}:{line_no}: invalid box coordinates"
                        ) from exc
                    if not (np.isfinite(nw) and np.isfinite(nh)) or nw <= 0 or nh <= 0:
                        raise ValueError(
                            f"{lbl_path}:{line_no}: non-positive box dimensions"
                        )
                    w_px = nw * w
                    h_px = nh * h
                    widths_px.append(w_px)
                    heights_px.append(h_px)
                    areas_px.append(w_px * h_px)
                    aspect_ratios.append(w_px / h_px)

    return pd.DataFrame({
        "pixel_width": widths_px,
        "pixel_height": heights_px,
        "pixel_area": areas_px,
        "aspect_ratio": aspect_ratios,
    })


def calculate_dataset_local_contrast(
    extracted_data_dir: Path,
    splits: list[str],
    *,
    selected_frames: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Computes Local Weber Contrast for all bounding boxes in positive frames across splits.
    Extracts target luminance vs surrounding background ring luminance.
    """
    data_records = []

    for split in splits:
        pos_dir = extracted_data_dir / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            continue

        for path in pos_dir.glob("*.txt"):
            if (
                selected_frames is not None
                and (
                    split,
                    path.with_suffix(".jpg").name,
                ) not in selected_frames
            ):
                continue
            meta = parse_frame_filename_logic(path.name)
            if meta["camera"] == "unknown" or meta["background"] == "unknown":
                continue

            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                test_path = path.with_suffix(ext)
                if test_path.exists():
                    img_path = test_path
                    break

            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            w_norm = float(parts[3])
                            h_norm = float(parts[4])
                            x_c = float(parts[1])
                            y_c = float(parts[2])

                            x1 = int((x_c - w_norm / 2.0) * orig_w)
                            y1 = int((y_c - h_norm / 2.0) * orig_h)
                            x2 = int((x_c + w_norm / 2.0) * orig_w)
                            y2 = int((y_c + h_norm / 2.0) * orig_h)

                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)

                            w_box = x2 - x1
                            h_box = y2 - y1

                            if w_box <= 0 or h_box <= 0:
                                continue

                            pad_w = int(w_box * 0.5)
                            pad_h = int(h_box * 0.5)

                            bg_x1 = max(0, x1 - pad_w)
                            bg_y1 = max(0, y1 - pad_h)
                            bg_x2 = min(orig_w - 1, x2 + pad_w)
                            bg_y2 = min(orig_h - 1, y2 + pad_h)

                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                            target_pixels = gray[y1:y2, x1:x2]
                            mean_target = np.mean(target_pixels)

                            bg_mask = np.zeros_like(gray[bg_y1:bg_y2, bg_x1:bg_x2])
                            rel_x1 = x1 - bg_x1
                            rel_y1 = y1 - bg_y1
                            rel_x2 = x2 - bg_x1
                            rel_y2 = y2 - bg_y1

                            bg_mask[:, :] = 255
                            bg_mask[rel_y1:rel_y2, rel_x1:rel_x2] = 0

                            bg_pixels = gray[bg_y1:bg_y2, bg_x1:bg_x2][bg_mask == 255]
                            mean_bg = np.mean(bg_pixels) if len(bg_pixels) > 0 else mean_target

                            contrast = abs(mean_target - mean_bg) / max(1.0, mean_bg)

                            data_records.append({
                                "camera": meta["camera"],
                                "background": meta["background"],
                                "contrast": contrast,
                            })

            except Exception as e:
                log.warning(f"Failed to analyze local contrast for {path.name}: {e}")

    return pd.DataFrame(data_records)