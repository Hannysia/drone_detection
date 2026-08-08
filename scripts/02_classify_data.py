import json
import logging
import os
import shutil
import time
from pathlib import Path
from google import genai
from tqdm import tqdm

from src.config import SOURCE_ROOT, CLASSIFIED_DIR, SAVE_DEBUG_FRAMES, VIDEO_EXTENSIONS, KEY_JSON_PATH
from src.processing.video import extract_keyframes_from_video
from src.processing.classifier import classify_single_video

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(KEY_JSON_PATH))

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main() -> None:
    with open(KEY_JSON_PATH, "r", encoding="utf-8") as f:

        key_data = json.load(f)
        project_id = key_data["project_id"]

    client = genai.Client(
        vertexai=True, 
        project=project_id, 
        location="us-central1"
    )

    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = os.listdir(CLASSIFIED_DIR)

    log.info(f"Starting to scan directory: {SOURCE_ROOT}...")

    all_videos = [
        p for p in SOURCE_ROOT.rglob("*") 
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    log.info(f"Videos found for processing: {len(all_videos)}")
    log.info(f"Previously processed files found: {len(existing_files)}\n" + "-"*40)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for video_path in tqdm(all_videos, desc="Video classification", unit="file"):
        current_folder_name = video_path.parent.name
        file_name_clean = video_path.stem
        file_ext = video_path.suffix.lower()
        
        expected_suffix = f"_{current_folder_name}_{file_name_clean}{file_ext}"
        already_processed = any(f.endswith(expected_suffix) for f in existing_files)
        
        if already_processed:
            skipped_count += 1
            continue
        
        try:
            frames = extract_keyframes_from_video(str(video_path))
            if not frames:
                tqdm.write(f" [!] Unable to extract frames: {video_path.name}")
                error_count += 1
                continue
                
            if SAVE_DEBUG_FRAMES:
                debug_dir = Path("debug_frames") / file_name_clean
                debug_dir.mkdir(parents=True, exist_ok=True)
                for i, pil_img in enumerate(frames):
                    pil_img.save(debug_dir / f"frame_{i:03d}.png")
                    
            classification = classify_single_video(client, frames)
            if not classification:
                tqdm.write(f" [!] Empty API response for: {video_path.name}")
                error_count += 1
                continue
                
            target = classification.get("target_shape", "unknown")
            camera = classification.get("camera_type", "unknown")
            bg = classification.get("background", "unknown")
            
            new_file_name = f"{target}_{camera}_{bg}_{current_folder_name}_{file_name_clean}{file_ext}"
            new_file_path = CLASSIFIED_DIR / new_file_name
            
            shutil.copy2(video_path, new_file_path)
            
            existing_files.append(new_file_name)
            processed_count += 1
            
            # Захист від перевищення квоти запитів API (Rate Limits)
            time.sleep(4.0)
            
        except Exception as e:
            tqdm.write(f" [CRITICAL ERROR] Failed file {video_path.name}: {e}")
            error_count += 1
            continue

    log.info("\n" + "=" * 40)
    log.info("Processing complete.")
    log.info(f"Successfully classified: {processed_count}")
    log.info(f"Skipped (already present): {skipped_count}")
    log.info(f"Errors/corrupted files: {error_count}")
    log.info("="*40)

if __name__ == "__main__":
    main()