import os
import re
import glob
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ================= CONFIGURATION =================
EXTRACTED_DATA_PATH = os.getenv("EXTRACTED_DATA_PATH", r"E:\backup\negatives")
WATERMARKS_DIR      = Path(r"E:\bachelor_data\watermarks")
SPLITS              = ["train", "val", "test"]
TARGET_SUBDIR       = "RAW_NEGATIVES"

PADDING             = 15    # Padding from frame borders in pixels

WATERMARK_FILES = {
    "wild_hornet": "WildHornet.png",
    "sbs":         "sbs.png",
    "signum":      "SIGNUM_black.png",
    "monomakh":    "mnmakh.png",
    "sternenko":   "Sternenko_Fund_Logo_Ukr.png",
    "prytula":     "prytula_fund.png"
}

# Individual scale factors optimized and verified by visual previews
WATERMARK_SCALES = {
    "wild_hornet": 0.40,  # Prominent centered lower watermark
    "sbs":         0.12,  # Baseline
    "signum":      0.15,  # Balanced for visual height
    "monomakh":    0.10,  # Balanced to match Signum
    "sternenko":   0.14,  # Increased by 15%
    "prytula":     0.14   # Increased by 15%
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

        # Apply specific watermark scales mapping
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


def process_splits():
    watermarks = load_watermarks(WATERMARKS_DIR, WATERMARK_FILES)

    print("=" * 80)
    print("RUNNING PRODUCTION WATERMARK OVERLAY PIPELINE")
    print("=" * 80)

    for split in SPLITS:
        split_dir = Path(EXTRACTED_DATA_PATH) / split / TARGET_SUBDIR
        if not split_dir.exists():
            print(f"Skipping split '{split}': directory not found: {split_dir}")
            continue

        image_files = sorted(glob.glob(str(split_dir / "*.jpg")))
        if not image_files:
            print(f"No JPEG frames found in {split_dir}")
            continue

        print(f"\nProcessing: {split.upper()} ({len(image_files):,} frames)")

        # -----------------------------------------------------------------
        # TRAIN SPLIT: Keep watermark layout consistent for all frames of a video
        # -----------------------------------------------------------------
        if split == "train":
            # Group files in E:\... by video ID
            video_groups = defaultdict(list)
            for filepath in image_files:
                stem = Path(filepath).stem
                video_id = re.sub(r'_f\d+$', '', stem)
                video_groups[video_id].append(filepath)

            print(f"  └─ Detected {len(video_groups)} unique videos in Train negatives.")
            
            for video_id, frames in tqdm(video_groups.items(), desc="Train Videos"):
                # Pick 1, 2, or 3 watermarks
                k = random.choice([1, 2, 3])
                combination = generate_combination(k)
                layout = assign_positions(combination)

                # Overwrite frames of this video identically
                for filepath in frames:
                    img = cv2.imread(filepath)
                    if img is None:
                        continue
                    img = apply_watermarks_to_frame(img, layout, watermarks)
                    cv2.imwrite(filepath, img)

        # -----------------------------------------------------------------
        # VAL & TEST SPLITS: Randomized on a frame-by-frame basis
        # -----------------------------------------------------------------
        else:
            for filepath in tqdm(image_files, desc=f"{split.capitalize()} Frames"):
                img = cv2.imread(filepath)
                if img is None:
                    continue
                
                k = random.choice([1, 2, 3])
                combination = generate_combination(k)
                layout = assign_positions(combination)

                img = apply_watermarks_to_frame(img, layout, watermarks)
                cv2.imwrite(filepath, img)

    print("\n" + "=" * 80)
    print("WATERMARK OVERLAY COMPLETED SUCCESSFULLY FOR ALL SPLITS")
    print("=" * 80)


if __name__ == "__main__":
    process_splits()