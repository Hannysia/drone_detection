import pytest
from src.processing.evaluation import evaluate_predictions, run_dataset_predictions_db
from unittest.mock import Mock
import cv2
import numpy as np

def test_evaluate_predictions_empty_preds():
    gt_boxes = [[100.0, 100.0, 200.0, 200.0]]
    pred_boxes = []
    
    tp, fp, fn = evaluate_predictions(gt_boxes, pred_boxes)
    
    assert tp == []
    assert fp == []
    assert fn == [0]


def test_evaluate_predictions_empty_gts():
    gt_boxes = []
    pred_boxes = [[100.0, 100.0, 200.0, 200.0, 0.9]]
    
    tp, fp, fn = evaluate_predictions(gt_boxes, pred_boxes)
    
    assert tp == []
    assert fp == [0]
    assert fn == []


def test_evaluate_predictions_perfect_match():
    gt_boxes = [[100.0, 100.0, 200.0, 200.0]]
    pred_boxes = [[100.0, 100.0, 200.0, 200.0, 0.9]]
    
    tp, fp, fn = evaluate_predictions(gt_boxes, pred_boxes)
    
    assert tp == [0]
    assert fp == []
    assert fn == []


def test_evaluate_predictions_greedy_matching_by_confidence():
    # One GT, two overlapping predictions.
    # The higher confidence pred (index 0) must match first.
    # The lower confidence pred (index 1) becomes a False Positive.
    gt_boxes = [[100.0, 100.0, 200.0, 200.0]]
    pred_boxes = [
        [110.0, 100.0, 210.0, 200.0, 0.95],  # IoU is ~0.81 (valid match)
        [100.0, 100.0, 200.0, 200.0, 0.80]   # IoU is 1.0 (but GT already matched)
    ]
    
    tp, fp, fn = evaluate_predictions(gt_boxes, pred_boxes)
    
    assert tp == [0]
    assert fp == [1]
    assert fn == []


def test_predictions_db_empty_directory(tmp_path):
    assert run_dataset_predictions_db(None, tmp_path, collection_conf=0.01) == []


def test_predictions_db_passes_collection_conf_to_model(tmp_path):
    folder = tmp_path / "RAW_NEGATIVES"
    folder.mkdir()
    assert cv2.imwrite(str(folder / "frame.png"), np.zeros((8, 8, 3), dtype=np.uint8))

    model = Mock()
    model.predict.return_value = [Mock(boxes=[])]

    rows = run_dataset_predictions_db(model, tmp_path, collection_conf=0.07)

    model.predict.assert_called_once()
    assert model.predict.call_args.kwargs["conf"] == 0.07
    assert [row["filename"] for row in rows] == ["frame.png"]
    assert rows[0]["gt_boxes"] == []



@pytest.mark.parametrize("category", ["RAW_POSITIVES", "RAW_NEGATIVES"])
def test_unreadable_image_stops_evaluation(tmp_path, category):
    folder = tmp_path / category
    folder.mkdir()
    image = folder / "broken.png"
    image.write_bytes(b"broken")
    model = Mock()

    with pytest.raises(ValueError) as error:
        run_dataset_predictions_db(model, tmp_path, collection_conf=0.07)

    assert str(image) in str(error.value)
    model.predict.assert_not_called()


@pytest.mark.parametrize(("content", "error_type"), [
    (None, FileNotFoundError),
    ("", ValueError),
    ("0 0.5 0.5 0.2 0.2\n0 0.5", ValueError),
    ("0 x 0.5 0.2 0.2", ValueError),
])
def test_invalid_positive_label_stops_evaluation(tmp_path, content, error_type):
    folder = tmp_path / "RAW_POSITIVES"
    folder.mkdir()
    image = folder / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))

    label = image.with_suffix(".txt")
    if content is not None:
        label.write_text(content, encoding="utf-8")

    model = Mock()
    with pytest.raises(error_type) as error:
        run_dataset_predictions_db(model, tmp_path, collection_conf=0.07)

    if isinstance(error.value, FileNotFoundError):
        assert error.value.filename == str(label)
    else:
        assert str(label) in str(error.value)
    model.predict.assert_not_called()