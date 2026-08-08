import logging
import shutil
import sys

from src.config import CLASSIFIED_DIR, DUPLICATES_DIR, HASH_THRESHOLD, VIDEO_EXTENSIONS
from src.processing.video import extract_video_hashes
from src.processing.deduplication import detect_video_duplicates

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main() -> None:
    if not CLASSIFIED_DIR.exists():
        log.error(f"Error: Classified directory does not exist: {CLASSIFIED_DIR}")
        sys.exit(1)
        
    DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)
    
    video_files = [
        f for f in CLASSIFIED_DIR.iterdir() 
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    
    log.info(f"🔍 Analyzing {len(video_files)} videos. Calculating hashes...")
    
    video_signatures = {}
    for v_path in video_files:
        try:
            video_signatures[v_path.name] = extract_video_hashes(v_path)
        except Exception as e:
            log.error(f"❌ Failed to extract hashes for {v_path.name}: {e}")
            continue
        
    log.info("⚔️ Comparing videos with each other...")
    duplicates_to_move = detect_video_duplicates(video_signatures, HASH_THRESHOLD)
            
    log.info("-" * 30)
    log.info(f"📂 Potential duplicates found: {len(duplicates_to_move)}")
    
    for dup in duplicates_to_move:
        src = CLASSIFIED_DIR / dup
        dst = DUPLICATES_DIR / dup
        try:
            shutil.move(str(src), str(dst))
            log.info(f"✅ Moved: {dup}")
        except OSError as e:
            log.error(f"❌ Error moving {dup}: {e}")
        
    log.info(f"✅ The deduplication process is complete. Duplicates have been moved to: '{DUPLICATES_DIR}'")

if __name__ == "__main__":
    main()