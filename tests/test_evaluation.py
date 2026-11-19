from unittest.mock import Mock
import cv2
import numpy as np
import pytest

from src.processing import evaluation
from src.processing.evaluation import (
    evaluate_model,
    evaluate_predictions,
    run_dataset_predictions_db,
)
from src.processing.manifests import CATEGORY_DIRS, write_manifest

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


def test_evaluate_predictions_returns_original_indices_when_unsorted():
    gt_boxes = [[100.0, 100.0, 200.0, 200.0]]
    pred_boxes = [
        [100.0, 100.0, 200.0, 200.0, 0.80],
        [110.0, 100.0, 210.0, 200.0, 0.95],
    ]

    tp, fp, fn = evaluate_predictions(gt_boxes, pred_boxes)

    assert tp == [1]
    assert fp == [0]
    assert fn == []


def _write_eval_bundle(tmp_path, rows, *, valid_images=True):
    root = tmp_path / "curated"
    for row in rows:
        folder = root / row["split"] / CATEGORY_DIRS[row["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        image = folder / row["filename"]
        if valid_images:
            assert cv2.imwrite(str(image), np.zeros((10, 10, 3), dtype=np.uint8))
            if row["category"] == "positive":
                image.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        else:
            image.write_bytes(b"broken")

    manifest_path = tmp_path / "main.csv"
    write_manifest(rows, manifest_path)
    return root, manifest_path


def test_predictions_db_rejects_train_split_and_empty_selection(tmp_path):
    root, manifest_path = _write_eval_bundle(
        tmp_path,
        [{"filename": "tr.jpg", "split": "train", "category": "negative", "excluded_at": ""}],
    )

    with pytest.raises(ValueError, match="not allowed"):
        run_dataset_predictions_db(
            Mock(),
            root,
            collection_conf=0.01,
            split="train",
            manifest_path=manifest_path,
        )

    with pytest.raises(ValueError, match="No active frames"):
        run_dataset_predictions_db(
            Mock(),
            root,
            collection_conf=0.01,
            split="val",
            manifest_path=manifest_path,
        )


def test_predictions_db_filters_by_manifest_and_passes_config(tmp_path):
    root, manifest_path = _write_eval_bundle(
        tmp_path,
        [
            {"filename": "tr_neg.jpg", "split": "train", "category": "negative", "excluded_at": ""},
            {"filename": "va_pos.jpg", "split": "val", "category": "positive", "excluded_at": ""},
            {"filename": "va_neg.jpg", "split": "val", "category": "negative", "excluded_at": ""},
            {"filename": "va_drop.jpg", "split": "val", "category": "negative", "excluded_at": "temporal"},
        ],
    )

    model = Mock()
    model.predict.return_value = [Mock(boxes=[])]

    rows = run_dataset_predictions_db(
        model,
        root,
        imgsz=1280,
        collection_conf=0.07,
        split="val",
        manifest_path=manifest_path,
    )

    assert model.predict.call_count == 2
    assert all(call.kwargs["conf"] == 0.07 and call.kwargs["imgsz"] == 1280 for call in model.predict.call_args_list)
    assert [row["filename"] for row in rows] == ["va_pos.jpg", "va_neg.jpg"]
    assert rows[0]["gt_boxes"][0] == pytest.approx([4.0, 4.0, 6.0, 6.0])
    assert rows[1]["gt_boxes"] == []


@pytest.mark.parametrize("category", ["positive", "negative"])
def test_unreadable_image_stops_evaluation(tmp_path, category):
    root, manifest_path = _write_eval_bundle(
        tmp_path,
        [{"filename": "broken.jpg", "split": "val", "category": category, "excluded_at": ""}],
        valid_images=False,
    )
    image = root / "val" / CATEGORY_DIRS[category] / "broken.jpg"
    model = Mock()

    with pytest.raises(ValueError) as error:
        run_dataset_predictions_db(
            model,
            root,
            collection_conf=0.07,
            split="val",
            manifest_path=manifest_path,
        )

    assert str(image) in str(error.value)
    model.predict.assert_not_called()


@pytest.mark.parametrize(("content", "error_type"), [
    (None, FileNotFoundError),
    ("", ValueError),
    ("0 0.5 0.5 0.2 0.2\n0 0.5", ValueError),
    ("0 x 0.5 0.2 0.2", ValueError),
])
def test_invalid_positive_label_stops_evaluation(tmp_path, content, error_type):
    root, manifest_path = _write_eval_bundle(
        tmp_path,
        [{"filename": "frame.jpg", "split": "val", "category": "positive", "excluded_at": ""}],
    )
    image = root / "val" / CATEGORY_DIRS["positive"] / "frame.jpg"
    label = image.with_suffix(".txt")
    if content is None:
        label.unlink()
    else:
        label.write_text(content, encoding="utf-8")

    model = Mock()
    with pytest.raises(error_type) as error:
        run_dataset_predictions_db(
            model,
            root,
            collection_conf=0.07,
            split="val",
            manifest_path=manifest_path,
        )

    if isinstance(error.value, FileNotFoundError):
        assert error.value.filename == str(label)
    else:
        assert str(label) in str(error.value)
    model.predict.assert_not_called()


def test_evaluate_model_uses_rect_false_imgsz_1280_and_auto_device(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\n", encoding="utf-8")
    output_dir = tmp_path / "runs"

    mock_yolo = Mock()
    mock_yolo.val.return_value = Mock(results_dict={"metrics/mAP50(B)": 0.91})
    monkeypatch.setattr(evaluation, "YOLO", Mock(return_value=mock_yolo))
    monkeypatch.setattr(evaluation, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: False)

    metrics = evaluate_model(weights, yaml_path, "run_with_negatives", conf=0.5)

    assert metrics == {"metrics/mAP50(B)": 0.91}
    mock_yolo.val.assert_called_once_with(
        data=str(yaml_path),
        split="test",
        imgsz=1280,
        batch=16,
        device="cpu",
        rect=False,
        project=str(output_dir),
        name="run_with_negatives_test_evaluation",
        exist_ok=True,
        save=True,
        plots=True,
        conf=0.5,
    )