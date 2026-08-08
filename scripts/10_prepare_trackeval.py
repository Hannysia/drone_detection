import logging
import sys

from src.config import LABELS_DIR, TRACKEVAL_DIR, VIDEO_EXTENSIONS
from src.processing.dataset import prepare_trackeval_sequence

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

SPLITS = ["test", "val"]

def main() -> None:
    if not LABELS_DIR.exists():
        log.error(f"Error: Base labels directory does not exist: {LABELS_DIR}")
        sys.exit(1)

    seqmaps_dir = TRACKEVAL_DIR / "seqmaps"
    seqmaps_dir.mkdir(parents=True, exist_ok=True)

    zip_files = {z.stem: z for z in LABELS_DIR.glob("*.zip")}

    for split in SPLITS:
        split_dir = LABELS_DIR / split
        if not split_dir.exists():
            log.warning(f"Warning: Split directory {split_dir} does not exist. Skipping.")
            continue
            
        log.info(f"\n--- Processing split: {split} ---")
        
        trackeval_split_name = f"drone-{split}"
        gt_dir = TRACKEVAL_DIR / trackeval_split_name
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        videos = [
            f for f in split_dir.iterdir() 
            if f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        matched_seqs = []
        
        for video_path in videos:
            seq_name = video_path.stem
            zip_path = zip_files.get(seq_name) or zip_files.get(f"{seq_name}_annotated")
            
            if zip_path:
                matched_seqs.append(seq_name)
                log.info(f"[{split.upper()}] Processing sequence: {seq_name}")
                
                seq_gt_dir = gt_dir / seq_name / "gt"
                seq_gt_dir.mkdir(parents=True, exist_ok=True)
                seqinfo_path = gt_dir / seq_name / "seqinfo.ini"
                
                try:
                    prepare_trackeval_sequence(
                        video_path=video_path,
                        zip_path=zip_path,
                        seq_gt_dir=seq_gt_dir,
                        seqinfo_path=seqinfo_path
                    )
                except Exception as e:
                    log.error(f"❌ Failed to process sequence {seq_name}: {e}")
            else:
                log.warning(f"Warning: No matching zip archive found for {video_path.name}")
                
        with open(seqmaps_dir / f"{trackeval_split_name}.txt", "w", encoding="utf-8") as f:
            f.write("name\n" + "\n".join(matched_seqs))

    log.info(f"\nDataset generation complete. Saved to: {TRACKEVAL_DIR}")

if __name__ == "__main__":
    main()