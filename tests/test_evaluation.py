from unittest.mock import Mock
import cv2
import numpy as np
import pytest

from src.processing import evaluation
from src.processing.evaluation import (
    analyze_model_errors,
    calibrate_confidence_thresholds,
    download_yolo_run_artifacts,
    evaluate_model,
    evaluate_predictions,
    evaluate_runs_on_test,
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


def test_download_yolo_run_artifacts_fetches_weights_and_csv(tmp_path, monkeypatch):
    def fake_download(repo_id, filename, repo_type, token, local_dir):
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ok")
        return str(target)

    monkeypatch.setattr(evaluation, "hf_hub_download", fake_download)
    artifacts = download_yolo_run_artifacts(
        ("run_positives_only", "run_with_negatives"),
        cache_dir=tmp_path,
        repo_id="user/ds",
        token="hf_tok",
    )

    assert artifacts["run_positives_only"]["weights"] == tmp_path / "models" / "yolo26n_run_positives_only_best.pt"
    assert artifacts["run_with_negatives"]["results_csv"] == tmp_path / "runs" / "run_with_negatives" / "results.csv"

    with pytest.raises(ValueError, match="Invalid run name"):
        download_yolo_run_artifacts(("../bad",), cache_dir=tmp_path, repo_id="user/ds", token="hf_tok")


def test_calibrate_confidence_thresholds_picks_max_f05_and_saves_plot(tmp_path):
    raw_database = [
        {
            "filename": "pos.jpg",
            "is_positive": True,
            "gt_boxes": [[0.0, 0.0, 10.0, 10.0]],
            "pred_boxes": [[0.0, 0.0, 10.0, 10.0, 0.85], [50.0, 50.0, 60.0, 60.0, 0.40]],
        },
        {
            "filename": "neg.jpg",
            "is_positive": False,
            "gt_boxes": [],
            "pred_boxes": [[0.0, 0.0, 5.0, 5.0, 0.40]],
        },
    ]

    cal_png = tmp_path / "cal.png"
    result = calibrate_confidence_thresholds(
        raw_database,
        thresholds=np.array([0.30, 0.60, 0.90]),
        save_path=cal_png,
        show=False,
    )

    assert result["opt_threshold_f05"] == pytest.approx(0.60)
    assert result["opt_threshold_f1"] == pytest.approx(0.60)
    assert cal_png.is_file()
    assert cal_png.stat().st_size > 0

    with pytest.raises(ValueError, match="empty"):
        calibrate_confidence_thresholds([])


def test_evaluate_runs_on_test_stages_evaluates_and_cleans_up(tmp_path, monkeypatch):
    root, manifest_path = _write_eval_bundle(
        tmp_path,
        [
            {"filename": "tr.jpg", "split": "train", "category": "positive", "excluded_at": ""},
            {"filename": "va.jpg", "split": "val", "category": "negative", "excluded_at": ""},
            {"filename": "te.jpg", "split": "test", "category": "positive", "excluded_at": ""},
        ],
    )
    w_pos = tmp_path / "pos.pt"
    w_neg = tmp_path / "neg.pt"
    w_pos.write_bytes(b"pt")
    w_neg.write_bytes(b"pt")

    working = tmp_path / "working"
    monkeypatch.setattr(evaluation.config, "WORKING_DIR", working)
    mock_eval = Mock(return_value={
        "metrics/precision(B)": 0.9,
        "metrics/recall(B)": 0.8,
        "metrics/mAP50(B)": 0.88,
        "metrics/mAP50-95(B)": 0.65,
    })
    monkeypatch.setattr(evaluation, "evaluate_model", mock_eval)

    table = evaluate_runs_on_test(
        {"run_positives_only": w_pos, "run_with_negatives": w_neg},
        root,
        calibrated_run="run_with_negatives",
        calibrated_conf=0.72,
        manifest_path=manifest_path,
    )

    assert list(table["Conf"]) == ["default (mAP)", "default (mAP)", "F0.5 (0.7200)"]
    assert mock_eval.call_count == 3
    assert mock_eval.call_args_list[2].kwargs["conf"] == pytest.approx(0.72)
    assert not (working / "yolo_dataset_eval_v1").exists()


def test_analyze_model_errors_breaks_down_scales_positions_metadata_and_modes():
    raw_database = [
        {
            "filename": "quad_rgb_sky_001_f0001.jpg",
            "is_positive": True,
            "width": 1000,
            "height": 1000,
            "gt_boxes": [
                [490.0, 490.0, 510.0, 510.0],
                [50.0, 50.0, 200.0, 200.0],
            ],
            "pred_boxes": [
                [490.0, 490.0, 510.0, 510.0, 0.90],
                [490.0, 490.0, 510.0, 510.0, 0.80],
                [50.0, 50.0, 200.0, 200.0, 0.40],
                [10.0, 10.0, 30.0, 30.0, 0.85],
            ],
        },
        {
            "filename": "fixed_thermal_forest_002_f0001.jpg",
            "is_positive": True,
            "width": 1000,
            "height": 1000,
            "gt_boxes": [[100.0, 100.0, 160.0, 160.0]],
            "pred_boxes": [[115.0, 115.0, 175.0, 175.0, 0.75]],
        },
        {
            "filename": "no_drone_rgb_sky_003_f0001.jpg",
            "is_positive": False,
            "width": 1000,
            "height": 1000,
            "gt_boxes": [],
            "pred_boxes": [[10.0, 10.0, 30.0, 30.0, 0.72]],
        },
    ]

    report = analyze_model_errors(raw_database, conf_threshold=0.70, top_k=2)

    modes = dict(zip(report["error_types"]["Type"], report["error_types"]["Count"]))
    assert modes == {
        "TP": 1,
        "Background FP": 2,
        "Localization FP": 1,
        "Duplicate FP": 1,
        "Threshold Miss FN": 1,
        "Blind Miss FN": 1,
    }

    by_size = report["by_size"].set_index("Scale")
    assert by_size.loc["small", "TP"] == 1
    assert by_size.loc["medium", "FN"] == 1
    assert by_size.loc["large", "FN"] == 1

    by_pos = report["by_position"].set_index("Region")
    assert by_pos.loc["Center", "Recall"] == pytest.approx(1.0)
    assert by_pos.loc["Off-center", "Recall"] == pytest.approx(0.0)

    by_bg = report["by_background"].set_index("Group")
    assert by_bg.loc["complex", "FN"] == 1
    assert by_bg.loc["sky", "Neg FA Rate (%)"] == pytest.approx(100.0)

    assert len(report["worst_fps"]) == 2
    assert len(report["worst_fns"]) == 2
    assert report["worst_fns"][0]["filename"] == "quad_rgb_sky_001_f0001.jpg"
    assert len(report["worst_fns"][0]["subthresh_boxes"]) == 1

    with pytest.raises(ValueError, match="empty"):
        analyze_model_errors([], conf_threshold=0.5)