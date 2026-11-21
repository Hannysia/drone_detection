from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.processing import adaptive_roi
from src.processing.diagnostics import letterbox_image


def _make_coord_frame(frame_idx: int, width: int = 1280, height: int = 720) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    xs = np.arange(width, dtype=np.int32)
    ys = np.arange(height, dtype=np.int32)
    x_high = np.clip(xs // 256, 0, 3)
    y_high = np.clip(ys // 256, 0, 3)
    img[:, :, 0] = (xs % 256)[None, :]
    img[:, :, 1] = (ys % 256)[:, None]
    img[:, :, 2] = ((frame_idx << 4) | (x_high[None, :] << 2) | y_high[:, None]).astype(np.uint8)
    return img


class _ExactCropDetector:
    def __init__(
        self,
        gt_centers: dict[int, tuple[float, float] | None],
        *,
        box_wh: tuple[float, float] = (32.0, 24.0),
        conf: float = 0.9,
        expect_crop: bool = False,
    ) -> None:
        self.gt_centers = gt_centers
        self.box_wh = box_wh
        self.conf = conf
        self.expect_crop = expect_crop
        self.calls: list[tuple[int, int, int, int | None, int | None]] = []

    def predict(self, bgr_image: np.ndarray, *, conf_thres: float | None = None) -> np.ndarray:
        h, w = bgr_image.shape[:2]
        meta = int(bgr_image[0, 0, 2])
        frame_idx = (meta >> 4) & 0x0F
        if self.expect_crop:
            x_high = (meta >> 2) & 0x03
            y_high = meta & 0x03
            x0 = int(bgr_image[0, 0, 0]) + x_high * 256
            y0 = int(bgr_image[0, 0, 1]) + y_high * 256
        else:
            x0, y0 = 0, 0

        self.calls.append((frame_idx, w, h, x0 if self.expect_crop else None, y0 if self.expect_crop else None))
        center = self.gt_centers.get(frame_idx)
        if center is None:
            return np.empty((0, 6), dtype=np.float32)

        cx, cy = center
        bw, bh = self.box_wh
        gx1, gy1, gx2, gy2 = cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
        if not self.expect_crop:
            return np.array([[gx1, gy1, gx2, gy2, self.conf, 0.0]], dtype=np.float32)

        lx1 = max(0.0, gx1 - x0)
        ly1 = max(0.0, gy1 - y0)
        lx2 = min(float(w), gx2 - x0)
        ly2 = min(float(h), gy2 - y0)
        if lx2 <= lx1 or ly2 <= ly1:
            return np.empty((0, 6), dtype=np.float32)
        return np.array([[lx1, ly1, lx2, ly2, self.conf, 0.0]], dtype=np.float32)


def test_decode_end2end_predictions_filters_and_rescales_from_letterbox():
    img = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    _, content_box = letterbox_image(img, imgsz=1280)
    top, bottom, left, right = content_box
    assert (top, bottom, left, right) == (280, 1000, 0, 1280)

    scale_x = (right - left) / 1920.0
    scale_y = (bottom - top) / 1080.0
    raw = np.array(
        [[
            [900.0 * scale_x + left, 600.0 * scale_y + top, 960.0 * scale_x + left, 630.0 * scale_y + top, 0.91, 0.0],
            [10.0, 300.0, 40.0, 340.0, 0.15, 0.0],
            [10.0, 300.0, 40.0, 340.0, 0.88, 1.0],
            [np.nan, 300.0, 40.0, 340.0, 0.95, 0.0],
        ]],
        dtype=np.float32,
    )

    detector = adaptive_roi.OpenVINOYOLODetector(
        imgsz=1280,
        conf_thres=0.25,
        runner=lambda tensor: raw,
    )
    preds = detector.predict(img)
    assert preds.shape == (1, 6)
    assert preds[0, :4] == pytest.approx([900.0, 600.0, 960.0, 630.0], abs=1e-2)
    assert preds[0, 4] == pytest.approx(0.91)
    assert preds[0, 5] == pytest.approx(0.0)


def test_roi_crop_window_clamping_and_global_coord_roundtrip():
    assert adaptive_roi.compute_roi_crop_window((960.0, 540.0), (1920, 1080), crop_size=640) == (640, 220, 1280, 860)
    assert adaptive_roi.compute_roi_crop_window((20.0, 15.0), (1920, 1080), crop_size=640) == (0, 0, 640, 640)
    assert adaptive_roi.compute_roi_crop_window((1910.0, 1070.0), (1920, 1080), crop_size=640) == (1280, 440, 1920, 1080)
    assert adaptive_roi.compute_roi_crop_window((100.0, 80.0), (320, 240), crop_size=640) == (0, 0, 320, 240)

    crop_box = (640, 220, 1280, 860)
    crop_det = np.array([[300.0, 310.0, 340.0, 330.0, 0.85, 0.0]], dtype=np.float32)
    global_det = adaptive_roi.map_crop_detections_to_global(crop_det, crop_box)
    assert global_det[0, :4] == pytest.approx([940.0, 530.0, 980.0, 550.0])
    assert adaptive_roi.map_crop_detections_to_global(np.empty((0, 6)), crop_box).shape == (0, 6)


def test_roi_forecaster_static_vs_kalman_and_large_gap_damping():
    static_fc = adaptive_roi.ROICenterForecaster(mode="static")
    static_fc.update((500.0, 300.0), dt_frames=1.0, frame_wh=(1280, 720))
    static_fc.update((520.0, 310.0), dt_frames=1.0, frame_wh=(1280, 720))
    static_fc.update(None, dt_frames=1.0, frame_wh=(1280, 720))
    assert static_fc.predict_center(1.0, frame_wh=(1280, 720)) == pytest.approx((520.0, 310.0))

    kalman_fc = adaptive_roi.ROICenterForecaster(mode="kalman")
    for idx in range(5):
        kalman_fc.update((400.0 + 20.0 * idx, 250.0 + 10.0 * idx), dt_frames=1.0, frame_wh=(1280, 720))

    pred_x, pred_y = kalman_fc.predict_center(1.0, frame_wh=(1280, 720))
    assert pred_x > 490.0
    assert pred_y > 294.0

    kalman_fc.update(None, dt_frames=1.0, frame_wh=(1280, 720))
    after_miss_x, after_miss_y = kalman_fc.predict_center(1.0, frame_wh=(1280, 720))
    assert after_miss_x > pred_x
    assert after_miss_y > pred_y

    gap_x, gap_y = kalman_fc.predict_center(50.0, frame_wh=(1280, 720))
    assert gap_x < after_miss_x + 30.0
    assert gap_y < after_miss_y + 30.0


@pytest.mark.parametrize("tracker_name", ["bytetrack", "ocsort"])
def test_adaptive_roi_runner_search_track_hysteresis_and_both_trackers(tmp_path, tracker_name):
    gt_centers = {
        1: None,
        2: (500.0, 350.0),
        3: (512.0, 354.0),
        4: (524.0, 358.0),
        5: (536.0, 362.0),
        6: None,
        7: None,
        8: None,
        9: (620.0, 390.0),
        10: (632.0, 394.0),
    }
    search_det = _ExactCropDetector(gt_centers, expect_crop=False)
    track_det = _ExactCropDetector(gt_centers, expect_crop=True)

    frames = [
        {"mot_frame": idx, "dt_frames": 1, "bgr_image": _make_coord_frame(idx, 1280, 720)}
        for idx in range(1, 11)
    ]
    tracker_cfg = (
        {"track_thresh": 0.5, "match_thresh": 0.8, "track_buffer": 10}
        if tracker_name == "bytetrack"
        else {"det_thresh": 0.5, "max_age": 10, "iou_threshold": 0.3}
    )
    runner = adaptive_roi.AdaptiveROIRunner(
        search_detector=search_det,
        track_detector=track_det,
        config=adaptive_roi.AdaptiveROIConfig(
            mode="roi_kalman",
            tracker_name=tracker_name,
            tracker_config=tracker_cfg,
            fps=30,
            max_lost_frames=2,
        ),
    )

    out_txt = tmp_path / f"seq_{tracker_name}.txt"
    result = runner.run_sequence(frames, output_txt_path=out_txt, return_frame_records=True)

    assert [c[0] for c in search_det.calls] == [1, 2, 9]
    assert [c[0] for c in track_det.calls] == [3, 4, 5, 6, 7, 8, 10]
    assert all((w, h) == (640, 640) for _, w, h, _, _ in track_det.calls)
    assert result["search_frames"] == 3
    assert result["track_frames"] == 7
    assert result["pct_track_frames"] == pytest.approx(70.0)
    assert result["search_to_track_transitions"] == 2
    assert result["track_to_search_transitions"] == 1
    assert result["e2e_mean_ms"] >= 0.0
    assert result["e2e_p95_ms"] >= result["e2e_p50_ms"]

    written = out_txt.read_text(encoding="utf-8").splitlines()
    min_rows = 4 if tracker_name == "bytetrack" else 3
    assert len(written) >= min_rows
    first_cols = written[0].split(",")
    assert len(first_cols) == 10
    assert [r["mode_used"] for r in result["frame_records"]] == [
        "search_1280",
        "search_1280",
        "track_640",
        "track_640",
        "track_640",
        "track_640",
        "track_640",
        "track_640",
        "search_1280",
        "track_640",
    ]


def test_run_adaptive_roi_on_sequences_writes_files_and_supports_all_four_modes(tmp_path):
    seq_dir = tmp_path / "frames"
    seq_dir.mkdir()
    frames = []
    gt_centers = {
        1: (400.0, 300.0),
        2: (412.0, 300.0),
        3: (424.0, 300.0),
        4: None,
        5: (448.0, 300.0),
    }
    for idx in range(1, 6):
        img_path = seq_dir / f"vid_001_f{idx:04d}.png"
        assert cv2.imwrite(str(img_path), _make_coord_frame(idx, 1280, 720))
        frames.append({"image_path": img_path, "mot_frame": idx, "dt_frames": 1})

    sequences = {"vid_001": frames}
    search_det = _ExactCropDetector(gt_centers, expect_crop=False)
    full_track_det = _ExactCropDetector(gt_centers, expect_crop=False)

    res_full = adaptive_roi.run_adaptive_roi_on_sequences(
        sequences,
        search_detector=search_det,
        config=adaptive_roi.AdaptiveROIConfig(mode="full_1280", tracker_name="bytetrack"),
        output_dir=tmp_path / "out_full",
    )
    assert res_full["pct_track_frames"] == pytest.approx(0.0)
    assert (tmp_path / "out_full" / "vid_001.txt").is_file()

    res_resize = adaptive_roi.run_adaptive_roi_on_sequences(
        sequences,
        track_detector=full_track_det,
        config=adaptive_roi.AdaptiveROIConfig(mode="resize_640", tracker_name="ocsort"),
        output_dir=tmp_path / "out_resize",
    )
    assert res_resize["pct_track_frames"] == pytest.approx(100.0)
    assert (tmp_path / "out_resize" / "vid_001.txt").is_file()

    crop_det_static = _ExactCropDetector(gt_centers, expect_crop=True)
    runner_static = adaptive_roi.AdaptiveROIRunner(
        search_detector=search_det,
        track_detector=crop_det_static,
        config=adaptive_roi.AdaptiveROIConfig(mode="roi_static", tracker_name="bytetrack", max_lost_frames=3),
    )
    out_static = runner_static.run_sequence(frames, return_frame_records=True)

    crop_det_kalman = _ExactCropDetector(gt_centers, expect_crop=True)
    runner_kalman = adaptive_roi.AdaptiveROIRunner(
        search_detector=search_det,
        track_detector=crop_det_kalman,
        config=adaptive_roi.AdaptiveROIConfig(mode="roi_kalman", tracker_name="bytetrack", max_lost_frames=3),
    )
    out_kalman = runner_kalman.run_sequence(frames, return_frame_records=True)

    static_crop_frame5 = out_static["frame_records"][4]["crop_box"]
    kalman_crop_frame5 = out_kalman["frame_records"][4]["crop_box"]
    assert static_crop_frame5[0] > 80
    assert kalman_crop_frame5[0] > static_crop_frame5[0]


def test_openvino_detector_compiles_once_with_latency_hint(tmp_path, monkeypatch):
    import openvino as ov

    with pytest.raises(FileNotFoundError):
        adaptive_roi.OpenVINOYOLODetector(tmp_path / "missing_dir", imgsz=640)

    ov_dir = tmp_path / "model_openvino_model"
    ov_dir.mkdir()
    (ov_dir / "best.xml").write_text("<net/>", encoding="utf-8")
    (ov_dir / "best.bin").write_bytes(b"weights")

    output_port = object()
    compiled_calls = []

    def fake_compiled(tensor):
        compiled_calls.append(tensor.shape)
        return {output_port: np.zeros((1, 0, 6), dtype=np.float32)}

    fake_compiled.output = Mock(return_value=output_port)
    fake_core = SimpleNamespace(
        read_model=Mock(return_value="ov_model"),
        compile_model=Mock(return_value=fake_compiled),
    )
    monkeypatch.setattr(ov, "Core", Mock(return_value=fake_core))

    detector = adaptive_roi.OpenVINOYOLODetector(ov_dir, imgsz=640)
    fake_core.compile_model.assert_called_once_with(
        "ov_model",
        device_name="CPU",
        config={"PERFORMANCE_HINT": "LATENCY"},
    )

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    assert detector.predict(frame).shape == (0, 6)
    assert detector.predict(frame).shape == (0, 6)
    assert len(compiled_calls) == 2
    assert fake_core.compile_model.call_count == 1