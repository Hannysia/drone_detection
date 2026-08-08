import argparse
import logging
import shutil
import sys
import torch
import numpy as np
from tqdm.auto import tqdm

from src.config import THETA_1_wv_pos, EXTRACTED_DATA_DIR, TRASH_DIR, EMBEDDINGS_DIR
from src.utils.io import move_to_trash
from src.processing.clustering import k_center_greedy_threshold
from src.processing.deduplication import group_files_by_video

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

SPLIT = "train"

def run_kcenter_deduplication(embedding_type: str) -> None:
    log.info(f"\nStage 2: Local k-Center Greedy Deduplication ({embedding_type.upper()})")
    log.info(f"Using distance threshold (θ₁): {THETA_1_wv_pos}")

    is_positive = (embedding_type == "positives")
    subdir_name = "RAW_POSITIVES" if is_positive else "RAW_NEGATIVES"
    
    target_data_dir = EXTRACTED_DATA_DIR / SPLIT / subdir_name
    if not target_data_dir.exists():
        log.error(f"[{SPLIT.upper()}] Directory not found: {target_data_dir}")
        return

    emb_file_name = f"{SPLIT}_{embedding_type}_embeddings.pt"
    emb_path = EMBEDDINGS_DIR / embedding_type / emb_file_name

    log.info(f"Loading embeddings from: {emb_path}")
    if not emb_path.exists():
        log.error(f"Embeddings file not found: {emb_path}")
        return

    try:
        embeddings_dict = torch.load(emb_path, map_location="cpu", weights_only=False)
    except Exception as e:
        log.error(f"Failed to load embeddings: {e}")
        return

    search_pattern = "*.txt" if is_positive else "*.jpg"
    all_files = sorted(list(target_data_dir.glob(search_pattern)))
    total_frames = len(all_files)

    video_groups = group_files_by_video(all_files)

    split_trash = TRASH_DIR / SPLIT / f"local_kcenter_{embedding_type}"
    if split_trash.exists():
        shutil.rmtree(split_trash)
    split_trash.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for video_name, files in tqdm(video_groups.items(), desc=f"Deduplicating {embedding_type}"):
        valid_files = []
        valid_embs = []
        
        for file_path in files:
            img_name = file_path.with_suffix(".jpg").name
            if img_name in embeddings_dict:
                valid_files.append(file_path)
                valid_embs.append(embeddings_dict[img_name])
        
        if len(valid_files) < 2:
            continue

        valid_embs = np.stack(valid_embs)

        if not is_positive:
            valid_embs = valid_embs[:, :768]

        if len(valid_embs) > 1000:
            log.warning(f"  [WARNING] Large video '{video_name}' detected: {len(valid_embs)} frames.")

        keep_indices = set(k_center_greedy_threshold(valid_embs, THETA_1_wv_pos))

        for idx, file_path in enumerate(valid_files):
            if idx not in keep_indices:
                try:
                    move_to_trash(file_path, split_trash, is_positive)
                    moved_count += 1
                except OSError as e:
                    log.error(e)

    log.info(
        f"\n[{SPLIT.upper()}] Deduplication complete."
        f"\nFiltered: {moved_count}/{total_frames} frames moved to trash."
        f"\nRemaining: {total_frames - moved_count}."
    )

if __name__ == "__main__":
    if not EXTRACTED_DATA_DIR or not TRASH_DIR or not EMBEDDINGS_DIR:
        log.error("Error: Environment variables not loaded. Make sure .env is configured.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(
        description="Run local k-Center deduplication for train split."
    )
    parser.add_argument(
        "-t", "--type",
        choices=["positives", "negatives"],
        required=True,
        help="Type of data to process: 'positives' or 'negatives'"
    )

    args = parser.parse_args()
    run_kcenter_deduplication(embedding_type=args.type)