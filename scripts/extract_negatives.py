import os
import sys
import cv2
from pathlib import Path
from dotenv import load_dotenv

# Setup paths to allow importing from 'src'
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables (.env)
load_dotenv()

# Read the destination directory from your .env configuration
EXTRACTED_DATA_PATH = Path(os.getenv("EXTRACTED_DATA_PATH"))

# Absolute local paths to your negative videos
NEGATIVE_VIDEOS_MAP = {
    "train": r"E:/bachelor_data/clean_classified/no_drone_rgb_226",  
    "val":   r"E:/bachelor_data/clean_classified/no_drone_rgb_225",      
    "test":  r"E:/bachelor_data/clean_classified/no_drone_rgb_224"     
}

NEGATIVE_VIDEOS_MAP = {
    "train": [
        r"E:\bachelor_data\negatives\no_drone_rgb_complex_597.mp4",
        r"E:\bachelor_data\negatives\no_drone_rgb_field_599.mp4",
        r"E:\bachelor_data\negatives\no_drone_rgb_sky_596.mp4",
        r"E:\bachelor_data\negatives\no_drone_thermal_complex_601.mp4" 
    ],
    "val": [
        r"E:\bachelor_data\negatives\no_drone_rgb_complex_600.mp4",
        r"E:\bachelor_data\negatives\no_drone_rgb_sky_598.mp4"
    ],
    "test": [
        r"E:\bachelor_data\negatives\no_drone_rgb_forest_595.mp4"
    ]
}

# Supported file extensions to try if none is explicitly provided
VIDEO_EXTENSIONS = ['.mp4', '._mp4', '.mov', '.m4v']
# =================================================

def extract_local_negative_video(split: str, base_video_path_str: str):
    """Reads a local video and extracts 100% of its frames directly to RAW_NEGATIVES."""
    target_neg_dir = EXTRACTED_DATA_PATH / split / "RAW_NEGATIVES"
    target_neg_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(base_video_path_str.strip())
    video_path = None

    # Resolve extension dynamically if not provided in the dictionary
    if base_path.exists() and base_path.is_file():
        video_path = base_path
    else:
        for ext in VIDEO_EXTENSIONS:
            test_path = base_path.with_suffix(ext)
            if test_path.exists():
                video_path = test_path
                break

    if not video_path:
        print(f"❌ ERROR: Video file not found for: {base_path}")
        print(f"   Checked extensions: {VIDEO_EXTENSIONS}")
        return

    video_base_name = video_path.stem
    print(f"\nProcessing local negative video: {video_base_name} -> {split.upper()}")
    print(f"  Source: {video_path}")
    print(f"  Destination: {target_neg_dir}")

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w == 0 or h == 0:
        print(f"  ⚠️ Error: Could not open video file: {video_path}")
        cap.release()
        return

    # Extract all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Format name (e.g., no_drone_rgb_226_f0001.jpg)
        img_name = f"{video_base_name}_f{frame_idx:04d}.jpg"
        target_img_path = target_neg_dir / img_name

        cv2.imwrite(str(target_img_path), frame)
        frame_idx += 1

    cap.release()
    print(f"  ✅ Successfully extracted {frame_idx - 1} frames to {target_neg_dir.name}")


def main():
    if not EXTRACTED_DATA_PATH.exists():
        print(f"Error: EXTRACTED_DATA_PATH does not exist: {EXTRACTED_DATA_PATH}")
        return

    for split, video_path in NEGATIVE_VIDEOS_MAP.items():
        extract_local_negative_video(split, video_path)

    print("\nLocal negative videos extraction finished successfully.")


if __name__ == "__main__":
    main()