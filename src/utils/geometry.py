def mot_to_yolo(
    box: list[float], 
    img_width: int, 
    img_height: int
) -> tuple[float, float, float, float]:
    """Converts MOT bounding box to YOLO normalized format."""
    x_left, y_top, w, h = box
    cx = (x_left + w / 2) / img_width
    cy = (y_top + h / 2) / img_height
    wn = w / img_width
    hn = h / img_height
    return cx, cy, wn, hn

def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Calculates IoU for two YOLO boxes (cx, cy, w, h normalized)."""
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    ix1, iy1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    ix2, iy2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    iarea = (ix2 - ix1) * (iy2 - iy1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    return iarea / (b1_area + b2_area - iarea + 1e-7)


def calculate_iou_xyxy(box1: list[float], box2: list[float]) -> float:
    """Calculates IoU for two absolute boxes in xyxy format [x_min, y_min, x_max, y_max]."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    iarea = (ix2 - ix1) * (iy2 - iy1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return iarea / (b1_area + b2_area - iarea + 1e-7)