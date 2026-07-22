import os
import glob
import cv2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# CONFIGURATION
# Set the crop percentages for each side (0.0 to 1.0)
# =====================================================================
TOP_CROP = 0.15     # 12% from the top
BOTTOM_CROP = 0.15  # 8% from the bottom
LEFT_CROP = 0.15    # 5% from the left
RIGHT_CROP = 0.15   # 5% from the right

EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH"))
SPLITS = ["train", "val", "test"]


def analyze_crop_impact():
    print(f"[*] Starting crop impact analysis with dynamic resolution resolution...")
    print(f"    Top: {TOP_CROP*100}%, Bottom: {BOTTOM_CROP*100}%")
    print(f"    Left: {LEFT_CROP*100}%, Right: {RIGHT_CROP*100}%")
    print("-" * 60)

    total_images_processed = 0
    total_original_boxes = 0
    total_boxes_lost = 0
    total_positives_turned_negatives = 0

    # Size-specific statistics (COCO categories mapped to absolute pixels)
    size_stats = {
        "Small": {"total": 0, "lost": 0},
        "Medium": {"total": 0, "lost": 0},
        "Large": {"total": 0, "lost": 0}
    }

    for split in SPLITS:
        positives_dir = os.path.join(EXTRACTED_DATA_DIR, split, "RAW_POSITIVES")
        if not os.path.exists(positives_dir):
            continue

        # Find all label files
        label_files = glob.glob(os.path.join(positives_dir, "*.txt"))
        for label_path in label_files:
            base_name = os.path.splitext(os.path.basename(label_path))[0]
            
            # Find the corresponding image file in the same folder to get exact dimensions
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                test_path = os.path.join(positives_dir, f"{base_name}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break

            if img_path is None:
                # If image is missing, we skip to maintain data integrity
                continue

            # Read image metadata/dimensions using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                continue

            orig_h, orig_w = map(float, img.shape[:2])
            total_images_processed += 1
            original_boxes_count = 0
            lost_boxes_count = 0

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                total_original_boxes += 1
                original_boxes_count += 1

                _, x_c, y_c, w, h = map(float, parts[:5])

                # Calculate original coordinates in original pixels
                x_min = x_c - (w / 2.0)
                y_min = y_c - (h / 2.0)
                x_max = x_c + (w / 2.0)
                y_max = y_c + (h / 2.0)

                # Classify original target size using absolute pixel area
                # Small < 1024 px (32x32), Medium < 9216 px (96x96), Large >= 9216 px
                orig_w_px = w * orig_w
                orig_h_px = h * orig_h
                area_pixels = orig_w_px * orig_h_px

                if area_pixels < 1024:
                    size_category = "Small"
                elif area_pixels < 9216:
                    size_category = "Medium"
                else:
                    size_category = "Large"

                size_stats[size_category]["total"] += 1

                # Calculate crop intersection boundaries in normalized space
                intersect_x_min = max(x_min, LEFT_CROP)
                intersect_y_min = max(y_min, TOP_CROP)
                intersect_x_max = min(x_max, 1.0 - RIGHT_CROP)
                intersect_y_max = min(y_max, 1.0 - BOTTOM_CROP)

                visible_w = max(0.0, intersect_x_max - intersect_x_min)
                visible_h = max(0.0, intersect_y_max - intersect_y_min)

                orig_area = w * h
                visible_area = visible_w * visible_h

                overlap_ratio = 0.0
                if orig_area > 0:
                    overlap_ratio = visible_area / orig_area

                # Check if box is lost (overlap < 70%)
                if overlap_ratio < 0.7 or visible_w <= 0 or visible_h <= 0:
                    total_boxes_lost += 1
                    lost_boxes_count += 1
                    size_stats[size_category]["lost"] += 1

            if original_boxes_count > 0 and lost_boxes_count == original_boxes_count:
                total_positives_turned_negatives += 1

    if total_images_processed == 0:
        print("[!] No matching positive images and labels found.")
        return

    box_loss_percentage = (total_boxes_lost / total_original_boxes * 100) if total_original_boxes > 0 else 0
    negative_conversion_pct = (total_positives_turned_negatives / total_images_processed * 100)

    print("============================================================")
    print("             DYNAMIC RESOLUTION CROP REPORT                 ")
    print("============================================================")
    print(f"Total Positive Images Analyzed      : {total_images_processed}")
    print(f"Total Original Bounding Boxes       : {total_original_boxes}")
    print(f"Total Bounding Boxes Lost           : {total_boxes_lost} ({box_loss_percentage:.2f}%)")
    print(f"Images Converted to Clean Negatives : {total_positives_turned_negatives} ({negative_conversion_pct:.2f}%)")
    print("-" * 60)
    print("LOSS BREAKDOWN BY OBJECT PHYSICAL SIZE:")
    for size, stats in size_stats.items():
        total_size = stats["total"]
        lost_size = stats["lost"]
        loss_pct = (lost_size / total_size * 100) if total_size > 0 else 0
        print(f"  - {size:<8} Objects: Total = {total_size:<5} | Lost = {lost_size:<5} ({loss_pct:.2f}%)")
    print("============================================================")


if __name__ == "__main__":
    analyze_crop_impact()