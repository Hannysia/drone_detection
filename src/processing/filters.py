import logging
from pathlib import Path

log = logging.getLogger(__name__)

def filter_label_file(lbl_path: Path, img_w: int, img_h: int, min_area: float) -> tuple[list[str], int]:
    """
    Parses a YOLO label file and filters out annotations smaller than the min_area.
    Returns:
        - List of preserved annotation strings
        - Number of deleted annotations
    """
    preserved_lines = []
    deleted_count = 0
    
    if not lbl_path.exists():
        return preserved_lines, deleted_count

    try:
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, cx, cy, nw, nh = map(float, parts)
            
            pixel_w = nw * img_w
            pixel_h = nh * img_h
            pixel_area = pixel_w * pixel_h
            
            if pixel_area >= min_area:
                preserved_lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            else:
                deleted_count += 1
                
    except Exception as e:
        log.error(f"Error parsing label file {lbl_path.name}: {e}")
        
    return preserved_lines, deleted_count