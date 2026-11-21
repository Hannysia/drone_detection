import importlib.util
from pathlib import Path
from unittest.mock import Mock

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pytest

from src.processing import video_ablation
from src.utils.visualization import (
    load_video_ablation_reports,
    plot_val_tracker_calibration,
    plot_video_ablation_dashboard,
)

matplotlib.use("Agg")


def _load_cli_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "07_video_ablation.py"
    spec = importlib.util.spec_from_file_location("video_ablation_cli_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_summary(run_dir: Path, *, hota: float, idf1: float, mota: float, idsw: int = 0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    headers = "HOTA DetA AssA LocA MOTA MOTP CLR_Re CLR_Pr CLR_TP CLR_FN CLR_FP IDSW Frag MT PT ML IDF1"
    values = f"{hota:.3f} 60.000 62.000 85.000 {mota:.3f} 84.000 78.000 90.000 10 2 1 {idf1:.3f} 1 1 0 0 {idf1:.3f}"
    (run_dir / "pedestrian_summary.txt").write_text(f"{headers}\n{values}\n", encoding="utf-8")


def _write_fake_sequence_bundle(tmp_path: Path, split: str, seq_name: str, n_frames: int = 4) -> dict[str, list[dict[str, object]]]:
    mot_dir = tmp_path / "mot_dataset"
    gt_dir = mot_dir / f"drone-{split}" / seq_name / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    seqinfo_dir = mot_dir / f"drone-{split}" / seq_name
    (seqinfo_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=30\nseqLength={n_frames}\nimWidth=1280\nimHeight=720\nimExt=.jpg\n",
        encoding="utf-8",
    )
    gt_lines = [f"{i},1,400.00,300.00,20.00,20.00,1,1,1.00" for i in range(1, n_frames + 1)]
    (gt_dir / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

    img_dir = tmp_path / "frames" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(1, n_frames + 1):
        img_path = img_dir / f"{seq_name}_f{i:04d}.jpg"
        assert cv2.imwrite(str(img_path), np.full((720, 1280, 3), 20 * i, dtype=np.uint8))
        frames.append({"video_id": seq_name, "image_path": img_path, "mot_frame": i, "dt_frames": 1})
    return {seq_name: frames}


class _FixedBoxDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, bgr_image: np.ndarray, *, conf_thres: float | None = None) -> np.ndarray:
        self.calls += 1
        h, w = bgr_image.shape[:2]
        if (w, h) == (640, 640):
            return np.array([[290.0, 290.0, 310.0, 310.0, 0.9, 0.0]], dtype=np.float32)
        return np.array([[400.0, 300.0, 420.0, 320.0, 0.9, 0.0]], dtype=np.float32)


def test_compute_track_small_recall_splits_coco_small_and_overall(tmp_path):
    gt_split_dir = tmp_path / "drone-test"
    tracker_data_dir = tmp_path / "tracker_data"
    (gt_split_dir / "seq1" / "gt").mkdir(parents=True)
    tracker_data_dir.mkdir(parents=True)

    (gt_split_dir / "seq1" / "gt" / "gt.txt").write_text(
        "\n".join([
            "1,1,10.00,10.00,10.00,10.00,1,1,1.00",
            "1,2,100.00,100.00,20.00,20.00,1,1,1.00",
            "2,1,200.00,200.00,50.00,50.00,1,1,1.00",
        ]),
        encoding="utf-8",
    )
    (tracker_data_dir / "seq1.txt").write_text(
        "\n".join([
            "1,1,10.00,10.00,10.00,10.00,0.95,-1,-1,-1",
            "1,2,500.00,500.00,15.00,15.00,0.60,-1,-1,-1",
            "2,1,200.00,200.00,50.00,50.00,0.92,-1,-1,-1",
        ]),
        encoding="utf-8",
    )

    metrics = video_ablation.compute_track_small_recall(gt_split_dir, tracker_data_dir, ["seq1"])
    assert metrics["gt_small_count"] == 2
    assert metrics["tp_small_count"] == 1
    assert metrics["track_recall_small"] == pytest.approx(0.5)
    assert metrics["gt_total_count"] == 3
    assert metrics["tp_total_count"] == 2
    assert metrics["track_recall_all"] == pytest.approx(2.0 / 3.0)
    assert metrics["track_precision_all"] == pytest.approx(2.0 / 3.0)


def test_calibrate_trackers_and_roi_on_val_selects_best_configs(tmp_path, monkeypatch):
    sequences = _write_fake_sequence_bundle(tmp_path, "val", "quad_001", n_frames=3)
    mot_dir = tmp_path / "mot_dataset"
    ablation_dir = tmp_path / "video_ablation"

    def fake_run_trackeval(split_name, trackers_to_eval, dataset_base_dir, trackers_out_dir):
        for name in trackers_to_eval:
            hota = 72.0 if ("run_1" in name or "lost4" in name) else 61.0
            _write_summary(
                Path(trackers_out_dir) / f"drone-{split_name}" / name,
                hota=hota,
                idf1=hota + 2.0,
                mota=hota - 1.0,
            )

    monkeypatch.setattr(video_ablation, "run_trackeval", fake_run_trackeval)
    det_1280 = _FixedBoxDetector()
    det_640 = _FixedBoxDetector()

    res = video_ablation.calibrate_trackers_and_roi_on_val(
        sequences,
        det_1280,
        det_640,
        mot_dir,
        ablation_dir,
        bytetrack_grid={"track_thresh": [0.4, 0.5], "track_buffer": [30], "match_thresh": [0.8]},
        ocsort_grid={"det_thresh": [0.4, 0.5], "max_age": [30], "iou_threshold": [0.3]},
        lost_frames_candidates=(3, 4),
    )

    assert (ablation_dir / "detections_cache" / "val" / "quad_001.txt").is_file()
    assert res["best_tracker_configs"]["bytetrack"]["track_thresh"] == 0.5
    assert res["best_tracker_configs"]["ocsort"]["det_thresh"] == 0.5
    assert res["best_max_lost_frames"] == 4
    assert res["val_grid_csv"].is_file()
    assert res["val_roi_csv"].is_file()


def test_run_video_ablation_on_test_supports_resume_and_visualizes(tmp_path, monkeypatch):
    sequences = _write_fake_sequence_bundle(tmp_path, "test", "quad_002", n_frames=3)
    mot_dir = tmp_path / "mot_dataset"
    ablation_dir = tmp_path / "video_ablation"

    def fake_run_trackeval(split_name, trackers_to_eval, dataset_base_dir, trackers_out_dir):
        for name in trackers_to_eval:
            hota = 68.0 if "roi_kalman" in name else 64.0
            _write_summary(
                Path(trackers_out_dir) / f"drone-{split_name}" / name,
                hota=hota,
                idf1=hota + 1.5,
                mota=hota - 2.0,
            )

    monkeypatch.setattr(video_ablation, "run_trackeval", fake_run_trackeval)
    det_1280 = _FixedBoxDetector()
    det_640 = _FixedBoxDetector()
    best_cfgs = {
        "bytetrack": {"track_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8},
        "ocsort": {"det_thresh": 0.5, "max_age": 30, "iou_threshold": 0.3},
    }

    res1 = video_ablation.run_video_ablation_on_test(
        sequences,
        det_1280,
        det_640,
        mot_dir,
        ablation_dir,
        tracker_configs=best_cfgs,
        max_lost_frames=4,
        modes=("full_1280", "roi_kalman"),
        trackers=("bytetrack", "ocsort"),
        eval_split="test",
    )
    assert len(res1["results_df"]) == 4
    assert res1["csv_path"].is_file()
    assert res1["metadata_path"].is_file()
    calls_after_first = (det_1280.calls, det_640.calls)

    res2 = video_ablation.run_video_ablation_on_test(
        sequences,
        det_1280,
        det_640,
        mot_dir,
        ablation_dir,
        tracker_configs=best_cfgs,
        max_lost_frames=4,
        modes=("full_1280", "roi_kalman"),
        trackers=("bytetrack", "ocsort"),
        eval_split="test",
        resume=True,
    )
    assert (det_1280.calls, det_640.calls) == calls_after_first
    assert len(res2["results_df"]) == 4

    loaded = load_video_ablation_reports(ablation_dir, eval_split="test")
    dash_path = tmp_path / "dash.png"
    plot_video_ablation_dashboard(loaded["results_df"], save_path=dash_path, show=False)
    assert dash_path.is_file()

    val_grid_df = pd.DataFrame([
        {"tracker": "bytetrack", "param_track_thresh": 0.4, "param_track_buffer": 30, "HOTA": 65.0},
        {"tracker": "bytetrack", "param_track_thresh": 0.5, "param_track_buffer": 30, "HOTA": 70.0},
        {"tracker": "ocsort", "param_det_thresh": 0.4, "param_max_age": 30, "HOTA": 62.0},
        {"tracker": "ocsort", "param_det_thresh": 0.5, "param_max_age": 30, "HOTA": 67.0},
    ])
    val_roi_df = pd.DataFrame([
        {"tracker": "bytetrack", "max_lost_frames": 3, "HOTA": 68.0},
        {"tracker": "bytetrack", "max_lost_frames": 4, "HOTA": 70.0},
        {"tracker": "ocsort", "max_lost_frames": 3, "HOTA": 65.0},
        {"tracker": "ocsort", "max_lost_frames": 4, "HOTA": 66.0},
    ])
    cal_path = tmp_path / "val_cal.png"
    plot_val_tracker_calibration(val_grid_df, val_roi_df=val_roi_df, save_path=cal_path, show=False)
    assert cal_path.is_file()


def test_video_ablation_cli_requires_val_metadata_before_test(tmp_path, monkeypatch):
    cli = _load_cli_module()
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "build_openvino_detector_pair", lambda *a, **k: (Mock(), Mock()))

    with pytest.raises(FileNotFoundError):
        cli.main(["--stage", "test", "--skip-prepare-gt", "--no-source-check"])