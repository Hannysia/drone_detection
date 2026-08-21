import pytest
from src.utils.geometry import mot_to_yolo, calculate_iou, calculate_iou_xyxy


def test_mot_to_yolo_standard():
    box = [100.0, 100.0, 50.0, 50.0]
    img_w, img_h = 640, 480

    cx, cy, w, h = mot_to_yolo(box, img_w, img_h)

    assert cx == pytest.approx(125.0 / 640)
    assert cy == pytest.approx(125.0 / 480)
    assert w == pytest.approx(50.0 / 640)
    assert h == pytest.approx(50.0 / 480)


def test_calculate_iou_identical():
    box = [0.5, 0.5, 0.2, 0.2]

    iou = calculate_iou(box, box)

    assert iou == pytest.approx(1.0, abs=1e-5)


def test_calculate_iou_no_overlap():
    box1 = [0.2, 0.2, 0.1, 0.1]
    box2 = [0.8, 0.8, 0.1, 0.1]

    iou = calculate_iou(box1, box2)

    assert iou == pytest.approx(0.0)


def test_calculate_iou_partial_overlap():
    box1 = [0.5, 0.5, 0.2, 0.2]      # limits: X [0.4, 0.6], Y [0.4, 0.6], Area = 0.04
    box2 = [0.55, 0.5, 0.2, 0.2]     # limits: X [0.45, 0.65], Y [0.4, 0.6], Area = 0.04
    # Intersection: X [0.45, 0.6] (width 0.15), Y [0.4, 0.6] (height 0.2) -> Area = 0.03
    # Union: 0.04 + 0.04 - 0.03 = 0.05
    # IoU: 0.03 / 0.05 = 0.6

    iou = calculate_iou(box1, box2)

    assert iou == pytest.approx(0.6, abs=1e-5)


def test_calculate_iou_xyxy_identical():
    box = [100.0, 100.0, 200.0, 200.0]

    iou = calculate_iou_xyxy(box, box)

    assert iou == pytest.approx(1.0)


def test_calculate_iou_xyxy_no_overlap():
    box1 = [100.0, 100.0, 200.0, 200.0]
    box2 = [300.0, 300.0, 400.0, 400.0]

    iou = calculate_iou_xyxy(box1, box2)

    assert iou == pytest.approx(0.0)


def test_calculate_iou_xyxy_partial_overlap():
    box1 = [100.0, 100.0, 200.0, 200.0]  # Width 100, Height 100 -> Area = 10,000
    box2 = [150.0, 100.0, 250.0, 200.0]  # Width 100, Height 100 -> Area = 10,000
    # Intersection: X [150, 200] (width 50), Y [100, 200] (height 100) -> Area = 5,000
    # Union: 10,000 + 10,000 - 5,000 = 15,000
    # IoU: 5,000 / 15,000 = 1/3 (approx 0.333333)

    iou = calculate_iou_xyxy(box1, box2)

    assert iou == pytest.approx(1.0 / 3.0)


def test_mot_to_yolo_zero_division():
    box = [100.0, 100.0, 50.0, 50.0]
    with pytest.raises(ZeroDivisionError):
        mot_to_yolo(box, 0, 480)


def test_calculate_iou_boundary_touch():
    box1 = [0.5, 0.5, 0.2, 0.2]
    box2 = [0.7, 0.5, 0.2, 0.2]
    
    iou = calculate_iou(box1, box2)
    
    assert iou == pytest.approx(0.0)