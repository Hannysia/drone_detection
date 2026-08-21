import pytest
from pathlib import Path
from src.processing.extractor import parse_annotations

def test_parse_annotations_standard_and_invalid_lines(tmp_path):
    mock_gt_file = tmp_path / "gt_mock.txt"
    
    gt_content = (
        "1,1,100.0,150.0,50.0,60.0,1,1,1\n"
        "1,2,200.0,250.0,30.0,40.0,1,1,1\n"
        "2,1,105.0,155.0,50.0,60.0,1,1,1\n"
        "3,1,10.0\n"
    )
    mock_gt_file.write_text(gt_content, encoding="utf-8")

    result = parse_annotations(mock_gt_file)

    assert len(result) == 2
    assert 1 in result
    assert 2 in result
    assert 3 not in result
    
    assert len(result[1]) == 2
    assert result[1][0] == [100.0, 150.0, 50.0, 60.0]
    assert result[1][1] == [200.0, 250.0, 30.0, 40.0]
    
    assert len(result[2]) == 1
    assert result[2][0] == [105.0, 155.0, 50.0, 60.0]