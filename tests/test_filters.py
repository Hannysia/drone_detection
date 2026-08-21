import pytest
from pathlib import Path
from src.processing.filters import filter_label_file

def test_filter_label_file_removes_micro_targets(tmp_path):
    mock_lbl_file = tmp_path / "label_mock.txt"
    
    # YOLO layout: class_id, cx, cy, nw, nh
    # Line 1: Drone (0), area on 1000x1000 img is 100x100 = 10,000 px² (Preserve)
    # Line 2: Drone (0), area on 1000x1000 img is 10x10 = 100 px² (Delete)
    # Line 3: Corrupted short line (must be skipped)
    lbl_content = (
        "0 0.500000 0.500000 0.100000 0.100000\n"
        "0 0.200000 0.200000 0.010000 0.010000\n"
        "0 0.10 0.10\n"
    )
    mock_lbl_file.write_text(lbl_content, encoding="utf-8")

    preserved, deleted = filter_label_file(
        lbl_path=mock_lbl_file,
        img_w=1000,
        img_h=1000,
        min_area=256.0
    )

    assert deleted == 1
    assert len(preserved) == 1
    assert preserved[0] == "0 0.500000 0.500000 0.100000 0.100000\n"