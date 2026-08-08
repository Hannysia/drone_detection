import re
from pathlib import Path
from src.utils.geometry import calculate_iou
from src.utils.io import read_yolo_bbox, move_to_trash
import itertools
import logging
import imagehash

log = logging.getLogger(__name__)

def group_labels_by_video(labels: list[Path]) -> dict[str, list[Path]]:
    """Groups label file paths by their video identifier."""
    groups = {}
    for lbl in labels:
        video_id = "_f".join(lbl.stem.split("_f")[:-1])
        groups.setdefault(video_id, []).append(lbl)
    
    for video_id in groups:
        groups[video_id].sort()
    return groups

def deduplicate_video_group(labels: list[Path], iou_threshold: float, trash_dir: Path) -> int:
    """Deduplicates a single sorted group of frames using IoU threshold. Returns count of removed frames."""
    if not labels:
        return 0

    moved_count = 0
    anchor_box = read_yolo_bbox(labels[0])

    for lbl_path in labels[1:]:
        try:
            curr_box = read_yolo_bbox(lbl_path)
            if calculate_iou(anchor_box, curr_box) > iou_threshold:
                move_to_trash(lbl_path, trash_dir, is_positive=True)
                moved_count += 1
            else:
                anchor_box = curr_box
        except (ValueError, IndexError) as e:
            continue

    return moved_count

def group_files_by_video(files: list[Path]) -> dict[str, list[Path]]:
    """Groups file paths by their video identifier using regex."""
    groups = {}
    for filepath in files:
        v_id = re.sub(r'_f\d+$', '', filepath.stem)
        groups.setdefault(v_id, []).append(filepath)
        
    for v_id in groups:
        groups[v_id].sort()
        
    return groups


def detect_video_duplicates(
    video_signatures: dict[str, list[imagehash.ImageHash]], 
    threshold: int
) -> set[str]:
    """Compares video perceptual signatures and returns a set of duplicate filenames."""
    duplicates_to_move = set()
    
    for vid_a, vid_b in itertools.combinations(video_signatures.keys(), 2):
        if vid_a in duplicates_to_move or vid_b in duplicates_to_move:
            continue
            
        hashes_a = video_signatures[vid_a]
        hashes_b = video_signatures[vid_b]
        
        is_duplicate = False
        for h_a in hashes_a:
            for h_b in hashes_b:
                if h_a - h_b <= threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                break
                
        if is_duplicate:
            log.info(f"⚠️ Знайдено дублікат: {vid_b} дуже схоже на {vid_a}")
            duplicates_to_move.add(vid_b)
            
    return duplicates_to_move