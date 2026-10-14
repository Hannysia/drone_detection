import logging
import shutil
import sys
import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm

from src.config import (
    RENAMED_DIR, SPLIT_OUTPUT_DIR, REPO_ID, VIDEO_EXTENSIONS, HF_TOKEN
)
from src.processing.video import get_frame_count
from src.processing.dataset import parse_filename, perform_stratified_split

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main() -> None:
    if not RENAMED_DIR.exists():
        log.error(f"Error: Source dataset directory does not exist: {RENAMED_DIR}")
        sys.exit(1)

    log.info(f"Scanning source directory: {RENAMED_DIR}")
    
    files = [
        f for f in RENAMED_DIR.iterdir() 
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    
    data = []
    for filepath in tqdm(files, desc="Reading video metadata"):
        combined_class = parse_filename(filepath.name)
        if combined_class:
            frames = get_frame_count(str(filepath))
            data.append({
                'filename': filepath.name, 
                'combined_class': combined_class, 
                'frames': frames
            })
                
    df = pd.DataFrame(data)
    if df.empty:
        log.error("Error: No valid video files found.")
        sys.exit(1)

    train_df, val_df, test_df = perform_stratified_split(df)

    total_frames = df['frames'].sum()
    log.info("\n📊 DISTRIBUTION OF RAW FRAMES BY SPLITS:")
    log.info(f"Train: {train_df['frames'].sum()} кадрів ({train_df['frames'].sum()/total_frames*100:.1f}%) | {len(train_df)} відео")
    log.info(f"Val:   {val_df['frames'].sum()} кадрів ({val_df['frames'].sum()/total_frames*100:.1f}%) | {len(val_df)} відео")
    log.info(f"Test:  {test_df['frames'].sum()} кадрів ({test_df['frames'].sum()/total_frames*100:.1f}%) | {len(test_df)} відео")

    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    log.info(f"\n[*] Preparing local directory structure in: {SPLIT_OUTPUT_DIR}")
    
    for split_name, split_data in splits.items():
        split_dir = SPLIT_OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for _, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Copying {split_name.upper()}"):
            src = RENAMED_DIR / row['filename']
            dst = split_dir / row['filename']
            shutil.copy2(src, dst)

    if HF_TOKEN and REPO_ID:
        log.info(f"\n🚀 Uploading the entire dataset to Hugging Face repository: {REPO_ID}...")
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(
                folder_path=str(SPLIT_OUTPUT_DIR),
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message="Initial commit: Stratified 70/15/15 split (Duration + Morphology)"
            )
            log.info("\n✅ Upload complete! Hugging Face now holds all three splits.")
        except Exception as e:
            log.error(f"❌ Failed to upload to Hugging Face: {e}")
    else:
        log.warning("\n⚠️ Skipping Hugging Face upload: HF_TOKEN or REPO_ID is missing in .env.")

if __name__ == "__main__":
    main()