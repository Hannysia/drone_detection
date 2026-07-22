import os
import re
import random
import cv2
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
# This is the NEW isolated directory that mirrors the original structure
NEW_EXTRACTED_DATA_PATH = r"E:\backup\extracted_dataset_new"
WATERMARKS_DIR          = Path(r"E:\bachelor_data\watermarks")

# Using the exact same subdirectory name as the original dataset for clean merging later
TARGET_SUBDIR           = "RAW_NEGATIVES"  

PADDING                 = 15    # Padding from frame borders in pixels

# Your 7 clean negative videos
NEGATIVE_VIDEOS_MAP = {
    "train": [
        r"E:\backup\negatives\no_drone_rgb_complex_597.mp4",
        r"E:\backup\negatives\no_drone_rgb_field_599.mp4",
        r"E:\backup\negatives\no_drone_rgb_sky_596.mp4",
        r"E:\backup\negatives\no_drone_thermal_complex_601.mp4"
    ],
    "val": [
        r"E:\backup\negatives\no_drone_rgb_complex_600.mp4",
        r"E:\backup\negatives\no_drone_rgb_sky_598.mp4"
    ],
    "test": [
        r"E:\backup\negatives\no_drone_rgb_forest_595.mp4"
    ]
}

WATERMARK_FILES = {
    "wild_hornet": "WildHornet.png",
    "sbs":         "sbs.png",
    "signum":      "SIGNUM_black.png",
    "monomakh":    "mnmakh.png",
    "sternenko":   "Sternenko_Fund_Logo_Ukr.png",
    "prytula":     "prytula_fund.png"
}

# Verified visual scales
WATERMARK_SCALES = {
    "wild_hornet": 0.40,
    "sbs":         0.12,
    "signum":      0.15,
    "monomakh":    0.10,
    "sternenko":   0.14,
    "prytula":     0.14
}
# =================================================


def load_watermarks(dir_path: Path, files_map: dict) -> dict:
    loaded = {}
    for key, filename in files_map.items():
        full_path = dir_path / filename
        if not full_path.exists():
            raise FileNotFoundError(f"Missing watermark PNG file: {full_path}")
        img = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] < 4:
            raise ValueError(f"Watermark must be a 4-channel transparent PNG: {full_path}")
        loaded[key] = img
    return loaded


def generate_combination(k: int) -> list[str]:
    scenarios = ["wild_hornet", "signum_only"]
    if k >= 2:
        scenarios.append("signum_monomakh")

    scenario = random.choice(scenarios)
    selected = []

    if scenario == "wild_hornet":
        selected.append("wild_hornet")
        pool = ["sbs", "sternenko", "prytula"]
        selected.extend(random.sample(pool, min(k - 1, len(pool))))
    elif scenario == "signum_only":
        selected.append("signum")
        pool = ["sbs", "sternenko", "prytula"]
        selected.extend(random.sample(pool, min(k - 1, len(pool))))
    elif scenario == "signum_monomakh":
        selected.extend(["signum", "monomakh"])
        pool = ["sbs", "sternenko", "prytula"]
        selected.extend(random.sample(pool, min(k - 2, len(pool))))

    return selected


def assign_positions(selected_keys: list[str]) -> dict[str, str]:
    corners = ["TL", "TR", "BL", "BR"]
    random.shuffle(corners)
    positions = {}

    if "wild_hornet" in selected_keys:
        positions["wild_hornet"] = "BC"

    if "signum" in selected_keys:
        pos = random.choice(["TL", "TR"])
        positions["signum"] = pos
        if pos in corners:
            corners.remove(pos)

    if "monomakh" in selected_keys:
        signum_pos = positions.get("signum")
        pos = "TR" if signum_pos == "TL" else "TL"
        positions["monomakh"] = pos
        if pos in corners:
            corners.remove(pos)

    for key in selected_keys:
        if key in ["wild_hornet", "signum", "monomakh"]:
            continue
        pos = corners.pop(0)
        positions[key] = pos

    return positions


def overlay_png(bg: np.ndarray, fg: np.ndarray, x: int, y: int, fg_w: int, fg_h: int) -> np.ndarray:
    h, w = bg.shape[:2]
    fg_resized = cv2.resize(fg, (fg_w, fg_h), interpolation=cv2.INTER_AREA)

    x1, x2 = max(0, x), min(w, x + fg_w)
    y1, y2 = max(0, y), min(h, y + fg_h)
    fg_x1 = max(0, -x)
    fg_y1 = max(0, -y)
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return bg

    bg_crop = bg[y1:y2, x1:x2]
    fg_crop = fg_resized[fg_y1:fg_y2, fg_x1:fg_x2]

    fg_rgb = fg_crop[:, :, :3]
    alpha = fg_crop[:, :, 3] / 255.0
    alpha = alpha[:, :, np.newaxis]

    blended = (1.0 - alpha) * bg_crop + alpha * fg_rgb
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg


def apply_watermarks_to_frame(frame: np.ndarray, layout: dict, watermarks: dict) -> np.ndarray:
    bg_h, bg_w = frame.shape[:2]

    for key, pos in layout.items():
        fg_img = watermarks[key]
        fg_h, fg_w = fg_img.shape[:2]

        scale = WATERMARK_SCALES.get(key, 0.12)
        target_w = int(bg_w * scale)
        target_h = int(fg_h * (target_w / fg_w))

        if pos == "TL":
            x, y = PADDING, PADDING
        elif pos == "TR":
            x, y = bg_w - target_w - PADDING, PADDING
        elif pos == "BL":
            x, y = PADDING, bg_h - target_h - PADDING
        elif pos == "BR":
            x, y = bg_w - target_w - PADDING, bg_h - target_h - PADDING
        elif pos == "BC":
            x, y = (bg_w - target_w) // 2, bg_h - target_h - PADDING
        else:
            continue

        frame = overlay_png(frame, fg_img, x, y, target_w, target_h)

    return frame


def run_pipeline():
    watermarks = load_watermarks(WATERMARKS_DIR, WATERMARK_FILES)

    print("=" * 80)
    print("STARTING PIPELINE: EXTRACT & WATERMARK TO NEW STAGING DATASET")
    print("=" * 80)

    for split, video_paths in NEGATIVE_VIDEOS_MAP.items():
        output_dir = Path(NEW_EXTRACTED_DATA_PATH) / split / TARGET_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing Split: {split.upper()} -> Saving to: {output_dir}")
        print("-" * 80)

        for video_path_str in video_paths:
            video_path = Path(video_path_str)
            if not video_path.exists():
                print(f"❌ File not found: {video_path_str}. Skipping...")
                continue

            video_name = video_path.stem
            print(f"Extracting & processing: {video_name}...")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"⚠️ Error opening video file: {video_path_str}")
                continue

            # For Train: lock configuration for the entire video
            if split == "train":
                k = random.choice([1, 2, 3])
                combination = generate_combination(k)
                layout = assign_positions(combination)

            frame_idx = 1
            extracted_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # For Val/Test: randomize frame-by-frame
                if split != "train":
                    k = random.choice([1, 2, 3])
                    combination = generate_combination(k)
                    layout = assign_positions(combination)

                processed_frame = apply_watermarks_to_frame(frame, layout, watermarks)

                # Save directly to E:\backup\extracted_dataset_new\[split]\RAW_NEGATIVES
                frame_filename = f"{video_name}_f{frame_idx:04d}.jpg"
                output_filepath = output_dir / frame_filename
                cv2.imwrite(str(output_filepath), processed_frame)

                frame_idx += 1
                extracted_count += 1

            cap.release()
            print(f"  └─ Success: Generated {extracted_count:,} watermarked frames for {video_name}")

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED! Please visually inspect frames in: {NEW_EXTRACTED_DATA_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()