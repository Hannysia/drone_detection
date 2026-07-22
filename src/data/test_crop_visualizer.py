import os
import glob
import cv2
import random
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# =====================================================================
# CONFIGURATION
# Set the strict binary crop percentages (0.0 to 1.0)
# =====================================================================
TOP_CROP = 0.15     # 22% to safely swallow top watermarks
BOTTOM_CROP = 0.15  # 22% to fully swallow bottom watermarks
LEFT_CROP = 0.15    # 22% to fully swallow left watermarks
RIGHT_CROP = 0.15   # 22% to fully swallow right watermarks

MIN_VISIBLE_PIXELS = 8.0  # Keep box if visible part is larger than 8x8 pixels

EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH"))
OUTPUT_DIR = "crop_verification"
NUM_TEST_IMAGES = 50


def process_test_crop():
    print(f"[*] Initializing visual verification script for {NUM_TEST_IMAGES} samples...")
    print(f"    Strict Binary Crop is ENABLED ('All-or-nothing' watermark protection).")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    positives_dir = os.path.join(EXTRACTED_DATA_DIR, "train", "RAW_POSITIVES")
    if not os.path.exists(positives_dir):
        print(f"[!] Path does not exist: {positives_dir}")
        return

    # Find matching image files (handling jpg and png cases)
    image_extensions = ["*.jpg", "*.png", "*.jpeg"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(positives_dir, ext)))

    if not image_paths:
        print("[!] No positive images found for visualization.")
        return

    # Take first 50 images
    random.seed(42)  # Keep selection deterministic
    selected_paths = image_paths[:NUM_TEST_IMAGES]
    processed_count = 0

    for img_path in selected_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(positives_dir, f"{base_name}.txt")

        if not os.path.exists(txt_path):
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        # Read original labels first to determine boundaries
        with open(txt_path, "r") as f:
            lines = f.readlines()

        # Define absolute threshold boundaries in pixel space
        top_limit_px = TOP_CROP * orig_h
        bottom_limit_px = (1.0 - BOTTOM_CROP) * orig_h
        left_limit_px = LEFT_CROP * orig_w
        right_limit_px = (1.0 - RIGHT_CROP) * orig_w

        # Crop flags (binary: True means full crop, False means 0% crop)
        crop_top = True
        crop_bottom = True
        crop_left = True
        crop_right = True

        bboxes_raw = []

        # Step 1: Detect if any bounding box violates the crop boundaries
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])

            # Convert to absolute pixel space on original image
            x_min = (x_c - w / 2.0) * orig_w
            y_min = (y_c - h / 2.0) * orig_h
            x_max = (x_c + w / 2.0) * orig_w
            y_max = (y_c + h / 2.0) * orig_h

            bboxes_raw.append((class_id, x_min, y_min, x_max, y_max))

            # Disable crop completely if a bounding box enters the respective exclusion zone
            if y_min < top_limit_px:
                crop_top = False
            if y_max > bottom_limit_px:
                crop_bottom = False
            if x_min < left_limit_px:
                crop_left = False
            if x_max > right_limit_px:
                crop_right = False

        # =====================================================================
        # STRICT BINARY LOGIC ("All-or-nothing" watermark protection):
        # If any single side is protected from crop (False), we disable crop
        # ENTIRELY for the whole image (all flags become False / 0% crop)
        # to guarantee no partially sliced watermarks can ever appear.
        # =====================================================================
        if not (crop_top and crop_bottom and crop_left and crop_right):
            crop_top = crop_bottom = crop_left = crop_right = False
        # =====================================================================

        # Step 2: Calculate binary crop boundaries (either full percentage crop or 0% crop)
        crop_x1 = int(left_limit_px) if crop_left else 0
        crop_y1 = int(top_limit_px) if crop_top else 0
        crop_x2 = int(right_limit_px) if crop_right else int(orig_w)
        crop_y2 = int(bottom_limit_px) if crop_bottom else int(orig_h)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        # Log dynamic binary decisions in the console
        print(f"[{processed_count + 1:02d}] {base_name:<30} | Crop status: "
              f"TOP={str(crop_top):<5} BOTTOM={str(crop_bottom):<5} "
              f"LEFT={str(crop_left):<5} RIGHT={str(crop_right):<5}")

        # Perform the dynamic dynamic crop
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        valid_boxes = []
        for class_id, x_min, y_min, x_max, y_max in bboxes_raw:
            # Shift and clamp boundaries
            new_x_min = max(0.0, min(x_min - crop_x1, float(crop_w)))
            new_y_min = max(0.0, min(y_min - crop_y1, float(crop_h)))
            new_x_max = max(0.0, min(x_max - crop_x1, float(crop_w)))
            new_y_max = max(0.0, min(y_max - crop_y1, float(crop_h)))

            new_w_px = new_x_max - new_x_min
            new_h_px = new_y_max - new_y_min

            if new_w_px >= MIN_VISIBLE_PIXELS and new_h_px >= MIN_VISIBLE_PIXELS:
                valid_boxes.append((class_id, new_x_min, new_y_min, new_x_max, new_y_max))

        # Step 3: Draw the recalculated boxes on the cropped visual
        for class_id, x1, y1, x2, y2 in valid_boxes:
            cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(cropped_img, f"Class: {class_id}", (int(x1), int(max(y1 - 5, 15))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save output image
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_cropped.jpg")
        cv2.imwrite(output_path, cropped_img)
        processed_count += 1

    print("-" * 60)
    print(f"[+] Visual test complete. Processed {processed_count} images.")
    print(f"    Check results in folder: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    process_test_crop()