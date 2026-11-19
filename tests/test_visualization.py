import inspect
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.utils import visualization
from src.utils.visualization import (
    load_yolo_training_history,
    overlay_heatmap,
    plot_detections_summary,
    plot_eigencam_pair_comparison,
    plot_false_positive_results,
    plot_unified_calibration_curves,
    plot_yolo_runs_comparison,
    plot_yolo_training_curves,
    summarize_yolo_training,
)


def _write_results_csv(path, rows, *, with_dfl=True, spaced_headers=False):
    headers = [
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "val/box_loss",
        "val/cls_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    if with_dfl:
        headers.extend(["train/dfl_loss", "val/dfl_loss"])

    sep = " , " if spaced_headers else ","
    lines = [sep.join(headers)]
    for row in rows:
        lines.append(",".join(str(row[col]) for col in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_yolo_training_history_strips_headers_and_sorts(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        [
            {
                "epoch": 2,
                "train/box_loss": 0.9,
                "train/cls_loss": 0.8,
                "val/box_loss": 1.0,
                "val/cls_loss": 0.9,
                "metrics/precision(B)": 0.85,
                "metrics/recall(B)": 0.78,
                "metrics/mAP50(B)": 0.82,
                "metrics/mAP50-95(B)": 0.56,
            },
            {
                "epoch": 1,
                "train/box_loss": 1.2,
                "train/cls_loss": 1.1,
                "val/box_loss": 1.3,
                "val/cls_loss": 1.2,
                "metrics/precision(B)": 0.70,
                "metrics/recall(B)": 0.65,
                "metrics/mAP50(B)": 0.68,
                "metrics/mAP50-95(B)": 0.42,
            },
        ],
        with_dfl=False,
        spaced_headers=True,
    )

    df = load_yolo_training_history(csv_path)
    assert list(df["epoch"]) == [1, 2]
    assert "train/box_loss" in df.columns


def test_load_yolo_training_history_rejects_invalid_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yolo_training_history(tmp_path / "missing.csv")

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("epoch,train/box_loss\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_yolo_training_history(empty_csv)

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recognized"):
        load_yolo_training_history(bad_csv)


def test_summarize_yolo_training_detects_overfitting_and_balanced(tmp_path):
    overfit_csv = tmp_path / "overfit.csv"
    _write_results_csv(
        overfit_csv,
        [
            {"epoch": 1, "train/box_loss": 1.5, "train/cls_loss": 1.4, "train/dfl_loss": 0.9, "val/box_loss": 1.5, "val/cls_loss": 1.4, "val/dfl_loss": 0.9, "metrics/precision(B)": 0.65, "metrics/recall(B)": 0.60, "metrics/mAP50(B)": 0.62, "metrics/mAP50-95(B)": 0.40},
            {"epoch": 2, "train/box_loss": 1.1, "train/cls_loss": 1.0, "train/dfl_loss": 0.7, "val/box_loss": 1.1, "val/cls_loss": 1.0, "val/dfl_loss": 0.7, "metrics/precision(B)": 0.85, "metrics/recall(B)": 0.80, "metrics/mAP50(B)": 0.84, "metrics/mAP50-95(B)": 0.60},
            {"epoch": 3, "train/box_loss": 0.8, "train/cls_loss": 0.7, "train/dfl_loss": 0.5, "val/box_loss": 1.3, "val/cls_loss": 1.2, "val/dfl_loss": 0.8, "metrics/precision(B)": 0.84, "metrics/recall(B)": 0.78, "metrics/mAP50(B)": 0.81, "metrics/mAP50-95(B)": 0.57},
            {"epoch": 4, "train/box_loss": 0.5, "train/cls_loss": 0.4, "train/dfl_loss": 0.3, "val/box_loss": 1.6, "val/cls_loss": 1.5, "val/dfl_loss": 1.0, "metrics/precision(B)": 0.82, "metrics/recall(B)": 0.75, "metrics/mAP50(B)": 0.78, "metrics/mAP50-95(B)": 0.54},
        ],
    )
    overfit_summary = summarize_yolo_training(load_yolo_training_history(overfit_csv))
    assert overfit_summary["best_epoch"] == 2
    assert overfit_summary["min_val_loss_epoch"] == 2
    assert overfit_summary["val_loss_rebound"] > 0
    assert overfit_summary["fit_status"] == "Overfitting signal"

    balanced_csv = tmp_path / "balanced.csv"
    _write_results_csv(
        balanced_csv,
        [
            {"epoch": 1, "train/box_loss": 1.4, "train/cls_loss": 1.3, "train/dfl_loss": 0.8, "val/box_loss": 1.5, "val/cls_loss": 1.4, "val/dfl_loss": 0.9, "metrics/precision(B)": 0.72, "metrics/recall(B)": 0.68, "metrics/mAP50(B)": 0.70, "metrics/mAP50-95(B)": 0.46},
            {"epoch": 2, "train/box_loss": 1.0, "train/cls_loss": 0.9, "train/dfl_loss": 0.6, "val/box_loss": 1.1, "val/cls_loss": 1.0, "val/dfl_loss": 0.7, "metrics/precision(B)": 0.89, "metrics/recall(B)": 0.82, "metrics/mAP50(B)": 0.88, "metrics/mAP50-95(B)": 0.65},
        ],
    )
    balanced_summary = summarize_yolo_training(load_yolo_training_history(balanced_csv))
    assert balanced_summary["best_epoch"] == 2
    assert balanced_summary["fit_status"] == "Balanced fit"


def test_plot_yolo_training_curves_saves_png_without_dfl(tmp_path):
    csv_path = tmp_path / "results.csv"
    out_png = tmp_path / "plots" / "training_curves.png"
    _write_results_csv(
        csv_path,
        [
            {"epoch": 1, "train/box_loss": 1.3, "train/cls_loss": 1.1, "val/box_loss": 1.4, "val/cls_loss": 1.2, "metrics/precision(B)": 0.75, "metrics/recall(B)": 0.70, "metrics/mAP50(B)": 0.73, "metrics/mAP50-95(B)": 0.50},
            {"epoch": 2, "train/box_loss": 0.9, "train/cls_loss": 0.8, "val/box_loss": 1.0, "val/cls_loss": 0.9, "metrics/precision(B)": 0.88, "metrics/recall(B)": 0.81, "metrics/mAP50(B)": 0.86, "metrics/mAP50-95(B)": 0.62},
        ],
        with_dfl=False,
    )

    summary = plot_yolo_training_curves(csv_path, "run_with_negatives", out_png, show=False)

    assert out_png.is_file()
    assert out_png.stat().st_size > 0
    assert summary["best_epoch"] == 2
    assert summary["best_map50"] == pytest.approx(0.86)


def test_plot_yolo_runs_comparison_returns_table_and_saves_png(tmp_path):
    csv_neg = tmp_path / "neg.csv"
    csv_pos = tmp_path / "pos.csv"
    out_png = tmp_path / "comparison.png"

    _write_results_csv(
        csv_neg,
        [
            {"epoch": 1, "train/box_loss": 1.2, "train/cls_loss": 1.1, "train/dfl_loss": 0.8, "val/box_loss": 1.3, "val/cls_loss": 1.2, "val/dfl_loss": 0.8, "metrics/precision(B)": 0.91, "metrics/recall(B)": 0.83, "metrics/mAP50(B)": 0.89, "metrics/mAP50-95(B)": 0.67},
        ],
    )
    _write_results_csv(
        csv_pos,
        [
            {"epoch": 1, "train/box_loss": 1.1, "train/cls_loss": 1.0, "train/dfl_loss": 0.7, "val/box_loss": 1.4, "val/cls_loss": 1.5, "val/dfl_loss": 0.9, "metrics/precision(B)": 0.78, "metrics/recall(B)": 0.84, "metrics/mAP50(B)": 0.80, "metrics/mAP50-95(B)": 0.59},
        ],
    )

    with pytest.raises(ValueError, match="at least one run"):
        plot_yolo_runs_comparison({}, out_png, show=False)

    table = plot_yolo_runs_comparison(
        {"With Negatives": csv_neg, "Positives Only": csv_pos},
        out_png,
        show=False,
    )

    assert out_png.is_file()
    assert out_png.stat().st_size > 0
    assert list(table["Run"]) == ["With Negatives", "Positives Only"]
    assert list(table["Precision"]) == pytest.approx([0.91, 0.78])


def test_visualization_has_no_processing_or_torch_imports():
    source = inspect.getsource(visualization)
    assert "src.processing" not in source
    assert "import torch" not in source


def test_plot_detections_summary_accepts_and_saves_output_path(tmp_path):
    out_png = tmp_path / "summary" / "detection_summary.png"
    plot_detections_summary([0, 1, 2], "run_with_negatives", out_png, show=False)
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_overlay_heatmap_unpads_letterbox_region_accurately():
    img_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    cam_map = np.zeros((40, 40), dtype=np.float32)
    cam_map[10:30, :] = 1.0

    overlaid = overlay_heatmap(
        img_rgb,
        cam_map,
        alpha=0.5,
        crop_box=(320, 960, 0, 1280),
        input_size=(1280, 1280),
    )
    assert overlaid.shape == img_rgb.shape
    assert overlaid.mean() > 20.0

    with pytest.raises(ValueError, match="both"):
        overlay_heatmap(img_rgb, cam_map, crop_box=(0, 10, 0, 10))


def test_plot_false_positive_results_saves_png(tmp_path):
    out_png = tmp_path / "fp_cam.png"
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    plot_false_positive_results(
        img,
        img,
        img,
        "neg_001.jpg",
        1,
        0.72,
        save_path=out_png,
        show=False,
    )
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_plot_eigencam_pair_comparison_and_calibration_save_png(tmp_path):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    pair_png = tmp_path / "pair.png"
    plot_eigencam_pair_comparison(
        img,
        img,
        img,
        "sample.jpg",
        "Positives Only",
        "With Negatives",
        save_path=pair_png,
        show=False,
    )
    assert pair_png.is_file()
    assert pair_png.stat().st_size > 0

    cal_png = tmp_path / "cal.png"
    plot_unified_calibration_curves(
        np.array([0.2, 0.5, 0.8]),
        [0.7, 0.9, 0.95],
        [0.9, 0.8, 0.5],
        [0.78, 0.85, 0.65],
        [0.73, 0.88, 0.81],
        [10.0, 2.0, 0.0],
        0.5,
        0.5,
        save_path=cal_png,
        show=False,
    )
    assert cal_png.is_file()
    assert cal_png.stat().st_size > 0