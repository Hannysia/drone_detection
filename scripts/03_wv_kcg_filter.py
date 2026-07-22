import os
import re
import sys
import glob
import shutil
import logging
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.config import THETA_1

SPLIT = "train"

EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH"))
TRASH_DIR = Path(os.getenv("TRASH_DIR"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR")) 

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def k_center_greedy_threshold(embeddings: np.ndarray, theta: float) -> list[int]:
    """
    Performs k-Center Greedy selection based on a distance threshold.
    Returns a list of indices representing the selected diverse frames.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    dists = 1.0 - sim

    selected_centers = [0]
    min_dists = dists[0].copy()

    while True:
        farthest_idx = np.argmax(min_dists)
        max_dist = min_dists[farthest_idx]

        if max_dist < theta:
            break

        selected_centers.append(farthest_idx)
        min_dists = np.minimum(min_dists, dists[farthest_idx])

    return selected_centers


def move_files_to_trash(label_path: Path, trash_dir: Path, is_positive: bool):
    """
    Moves files to the specific trash directory.
    For positives, we move both .txt and .jpg.
    For negatives, we move only .jpg (negatives don't have .txt labels).
    """
    try:
        img_path = label_path.with_suffix('.jpg')
        
        if is_positive:
            if label_path.exists():
                shutil.move(str(label_path), trash_dir / label_path.name)
        
        if img_path.exists():
            shutil.move(str(img_path), trash_dir / img_path.name)
            
    except OSError as e:
        log.error(f"Error moving file {label_path.name} to trash: {e}")


def run_kcenter_deduplication(embedding_type: str):
    log.info(f"\nStage 2: Local k-Center Greedy Deduplication ({embedding_type.upper()})")
    log.info(f"Using distance threshold (θ₁): {THETA_1}")

    # Set dynamic subfolders based on the execution type
    is_positive = (embedding_type == "positives")
    subdir_name = "RAW_POSITIVES" if is_positive else "RAW_NEGATIVES"
    
    target_data_dir = EXTRACTED_DATA_DIR / SPLIT / subdir_name
    if not target_data_dir.exists():
        log.error(f"[{SPLIT.upper()}] Directory not found: {target_data_dir}")
        return

    # Dynamic local file path matching the clean download structure
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

    # For positives we scan .txt files, for negatives we scan .jpg files directly
    search_pattern = "*.txt" if is_positive else "*.jpg"
    all_files = sorted(glob.glob(str(target_data_dir / search_pattern)))
    total_frames = len(all_files)

    # Group files by video ID using robust regex splitting
    video_groups = {}
    for filepath_str in all_files:
        filepath = Path(filepath_str)
        # Safely remove only the final '_fXXXX' (frame index) from the name
        v_id = re.sub(r'_f\d+$', '', filepath.stem)
        video_groups.setdefault(v_id, []).append(filepath)
        
    for v_id in video_groups:
        video_groups[v_id].sort()

    # Re-create trash folder specific to this type (e.g., local_kcenter_positives)
    split_trash = TRASH_DIR / SPLIT / f"local_kcenter_{embedding_type}"
    if split_trash.exists():
        shutil.rmtree(split_trash)
    split_trash.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    # Run k-Center selection per video
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
            log.warning(f"  [WARNING] Large video '{video_name}' detected: {len(valid_embs)} frames. Processing might be slow.")

        keep_indices = set(k_center_greedy_threshold(valid_embs, THETA_1))

        for idx, file_path in enumerate(valid_files):
            if idx not in keep_indices:
                move_files_to_trash(file_path, split_trash, is_positive)
                moved_count += 1

    log.info(
        f"\n[{SPLIT.upper()}] Deduplication complete."
        f"\nFiltered: {moved_count}/{total_frames} frames moved to trash."
        f"\nRemaining: {total_frames - moved_count}."
    )


if __name__ == "__main__":
    if not os.getenv("EXTRACTED_DATA_PATH") or not os.getenv("TRASH_DIR") or not os.getenv("EMBEDDINGS_DIR"):
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