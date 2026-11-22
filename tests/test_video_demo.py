import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.processing import video_demo


def _write_fake_openvino_dir(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.xml").write_text("<net/>", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"bin")
    return model_dir


def _write_synthetic_video(path: Path, *, width: int = 960, height: int = 720, n_frames: int = 6, fps: float = 15.0) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for idx in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        cx = 240 + idx * 45
        cy = 320 + idx * 12
        cv2.rectangle(frame, (cx - 18, cy - 18), (cx + 18, cy + 18), (240, 240, 240), -1)
        writer.write(frame)
    writer.release()
    return path


class _WhiteBoxDetector:
    def predict(self, bgr_image: np.ndarray, *, conf_thres: float | None = None) -> np.ndarray:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 200)
        if xs.size == 0 or ys.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.array(
            [[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), 0.92, 0.0]],
            dtype=np.float32,
        )


def test_resolve_demo_openvino_models_handles_explicit_local_and_hf_fallback(tmp_path, monkeypatch):
    explicit_1280 = _write_fake_openvino_dir(tmp_path / "exp_1280")
    explicit_640 = _write_fake_openvino_dir(tmp_path / "exp_640")
    mock_dl = Mock()
    monkeypatch.setattr(video_demo, "snapshot_download", mock_dl)

    assert video_demo.resolve_demo_openvino_models(
        model_1280_dir=explicit_1280,
        model_640_dir=explicit_640,
    ) == (explicit_1280, explicit_640)
    mock_dl.assert_not_called()

    local_base = tmp_path / "local_base"
    local_1280 = _write_fake_openvino_dir(
        local_base / "working" / "quantization" / "1280" / f"{video_demo.YOLO_MODEL_STEM}_int8_1280_openvino_model"
    )
    local_640 = _write_fake_openvino_dir(
        local_base / "working" / "quantization" / "640" / f"{video_demo.YOLO_MODEL_STEM}_int8_640_openvino_model"
    )
    monkeypatch.setenv("LOCAL_BASE_DIR", str(local_base))
    assert video_demo.resolve_demo_openvino_models() == (local_1280, local_640)
    mock_dl.assert_not_called()

    monkeypatch.delenv("LOCAL_BASE_DIR", raising=False)
    snap_dir = tmp_path / "hf_snap"
    hf_1280 = _write_fake_openvino_dir(
        snap_dir / "models" / "quantization" / "1280" / f"{video_demo.YOLO_MODEL_STEM}_int8_1280_openvino_model"
    )
    hf_640 = _write_fake_openvino_dir(
        snap_dir / "models" / "quantization" / "640" / f"{video_demo.YOLO_MODEL_STEM}_int8_640_openvino_model"
    )
    mock_hf = Mock(return_value=str(snap_dir))
    monkeypatch.setattr(video_demo, "snapshot_download", mock_hf)

    got_1280, got_640 = video_demo.resolve_demo_openvino_models(repo_id="user/drone-ds", token="tok")
    assert (got_1280, got_640) == (hf_1280, hf_640)
    mock_hf.assert_called_once()
    assert mock_hf.call_args.kwargs["repo_type"] == "dataset"


def test_render_demo_frame_draws_roi_and_resets_trail_on_id_switch():
    base = np.zeros((480, 640, 3), dtype=np.uint8)
    trail = video_demo.TrailState(maxlen=10)
    rec_1 = {
        "mot_frame": 1,
        "mode_used": "track_640",
        "crop_box": (40, 40, 300, 300),
        "lost_frames": 0,
        "target_track_id": 1,
        "detections": [],
        "tracks": [[100.0, 100.0, 140.0, 140.0, 1.0, 0.9]],
        "latency_ms": 20.0,
    }
    rec_2 = {
        **rec_1,
        "mot_frame": 2,
        "target_track_id": 2,
        "tracks": [[200.0, 200.0, 240.0, 240.0, 2.0, 0.88]],
    }

    with_roi = video_demo.render_demo_frame(base, rec_1, mode="roi_kalman", trail_state=trail, draw_roi_box=True, draw_trail=True)
    without_roi = video_demo.render_demo_frame(base, rec_1, mode="roi_kalman", trail_state=None, draw_roi_box=False, draw_trail=False)
    assert np.any(base == 0) and not np.array_equal(with_roi, without_roi)
    assert len(trail.points) == 1

    video_demo.render_demo_frame(base, rec_2, mode="roi_kalman", trail_state=trail, draw_roi_box=True, draw_trail=True)
    assert trail.active_track_id == 2
    assert len(trail.points) == 1


@pytest.mark.parametrize("mode", ["roi_kalman", "roi_static", "full_1280", "resize_640"])
def test_process_video_file_streams_modes_and_writes_mp4(tmp_path, mode):
    in_video = _write_synthetic_video(tmp_path / f"input_{mode}.mp4", width=960, height=720, n_frames=6)
    out_video = tmp_path / f"out_{mode}.mp4"
    det = _WhiteBoxDetector()

    result = video_demo.process_video_file(
        in_video,
        output_video_path=out_video,
        mode=mode,
        tracker_name="bytetrack",
        tracker_config={"track_thresh": 0.3, "match_thresh": 0.8, "track_buffer": 15},
        max_lost_frames=3,
        search_conf=0.1,
        track_conf=0.1,
        seed_conf=0.25,
        search_detector=det,
        track_detector=det,
    )

    assert out_video.is_file() and out_video.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_video))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded += 1
        assert frame.shape[:2] == (720, 960)
    cap.release()
    assert decoded == 6

    summary = result["summary"]
    df = result["per_frame_df"]
    assert summary["total_frames"] == 6
    assert len(df) == 6
    if mode in ("roi_kalman", "roi_static"):
        assert summary["search_frames"] >= 1
        assert summary["track_frames"] >= 1
        assert df["has_roi_crop"].any()
    elif mode == "full_1280":
        assert summary["search_frames"] == 6
        assert summary["track_frames"] == 0
    else:
        assert summary["search_frames"] == 0
        assert summary["track_frames"] == 6


def test_process_video_file_validates_inputs_and_max_frames(tmp_path):
    in_video = _write_synthetic_video(tmp_path / "clip.mp4", width=640, height=480, n_frames=5)
    det = _WhiteBoxDetector()

    with pytest.raises(ValueError, match="Unsupported demo mode"):
        video_demo.process_video_file(in_video, mode="bad_mode", search_detector=det, track_detector=det)

    with pytest.raises(FileNotFoundError, match="Input video not found"):
        video_demo.process_video_file(tmp_path / "missing.mp4", search_detector=det, track_detector=det)

    res = video_demo.process_video_file(
        in_video,
        output_video_path=tmp_path / "capped.mp4",
        mode="full_1280",
        max_frames=3,
        search_detector=det,
    )
    assert res["summary"]["total_frames"] == 3
    assert len(res["per_frame_df"]) == 3



def _load_app_for_test(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    try:
        importlib.import_module("gradio")
    except ModuleNotFoundError:
        fake_gr = ModuleType("gradio")

        class _FakeError(Exception):
            pass

        class _FakeWidget:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def click(self, *args, **kwargs):
                return None

        fake_gr.Error = _FakeError
        for name in (
            "Blocks",
            "Markdown",
            "Row",
            "Column",
            "Video",
            "Radio",
            "Dropdown",
            "Accordion",
            "Slider",
            "Checkbox",
            "Button",
            "Plot",
            "Dataframe",
            "Progress",
        ):
            setattr(fake_gr, name, _FakeWidget)
        monkeypatch.setitem(sys.modules, "gradio", fake_gr)

    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_app_run_demo_inference_caches_detectors_and_builds_telemetry(tmp_path, monkeypatch):
    import matplotlib.figure
    import pandas as pd

    app = _load_app_for_test(monkeypatch)
    app._DETECTOR_CACHE.clear()
    det = _WhiteBoxDetector()
    load_mock = Mock(return_value=(det, det))
    monkeypatch.setattr(app, "build_demo_detector_pair", load_mock)

    with pytest.raises(app.gr.Error, match="Upload a video"):
        app.run_demo_inference(None, "roi_kalman", "bytetrack", 4, 0.1, 0.1, 0.25, 0, True, True, True)

    in_video = _write_synthetic_video(tmp_path / "space_clip.mp4", width=960, height=720, n_frames=5)
    progress_calls = []

    out_path_1, summary_df_1, fig_1 = app.run_demo_inference(
        {"path": str(in_video)},
        "roi_kalman",
        "bytetrack",
        3,
        0.1,
        0.1,
        0.25,
        0,
        True,
        True,
        True,
        progress=lambda step, desc="": progress_calls.append((step, desc)),
    )
    out_path_2, _, _ = app.run_demo_inference(
        str(in_video),
        "full_1280",
        "bytetrack",
        3,
        0.1,
        0.1,
        0.25,
        2,
        False,
        False,
        True,
    )

    assert load_mock.call_count == 1
    assert Path(out_path_1).is_file() and Path(out_path_2).is_file()
    assert isinstance(summary_df_1, pd.DataFrame) and not summary_df_1.empty
    assert isinstance(fig_1, matplotlib.figure.Figure) and len(fig_1.axes) == 2
    assert len(progress_calls) == 5
    assert app.build_demo_app() is not None



def test_resolve_demo_openvino_models_falls_back_to_hf_manifest_without_revision(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_BASE_DIR", raising=False)
    monkeypatch.delenv("REPO_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(video_demo.config, "REPO_ID", None, raising=False)
    monkeypatch.setattr(video_demo.config, "HF_TOKEN", None, raising=False)

    manifests_dir = tmp_path / "manifests"
    hf_source = manifests_dir / "sources" / "hf.json"
    hf_source.parent.mkdir(parents=True, exist_ok=True)
    hf_source.write_text(
        json.dumps({"repo_id": "owner/ds", "repo_type": "dataset", "revision": "old-rev"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(video_demo.config, "MANIFESTS_DIR", manifests_dir, raising=False)

    snap_dir = tmp_path / "hf_snap"
    hf_1280 = _write_fake_openvino_dir(
        snap_dir / "models" / "quantization" / "1280" / f"{video_demo.YOLO_MODEL_STEM}_int8_1280_openvino_model"
    )
    hf_640 = _write_fake_openvino_dir(
        snap_dir / "models" / "quantization" / "640" / f"{video_demo.YOLO_MODEL_STEM}_int8_640_openvino_model"
    )
    mock_hf = Mock(return_value=str(snap_dir))
    monkeypatch.setattr(video_demo, "snapshot_download", mock_hf)

    got_1280, got_640 = video_demo.resolve_demo_openvino_models()
    assert (got_1280, got_640) == (hf_1280, hf_640)
    assert mock_hf.call_args.kwargs["repo_id"] == "owner/ds"
    assert mock_hf.call_args.kwargs["repo_type"] == "dataset"
    assert "revision" not in mock_hf.call_args.kwargs

    with pytest.raises(ValueError, match="REPO_ID is required"):
        video_demo.resolve_demo_openvino_models(source_path=tmp_path / "missing.json")


def test_create_mp4_writer_prefers_h264_before_mp4v(tmp_path, monkeypatch):
    attempted = []
    released = []

    class _FakeWriter:
        def __init__(self, codec: str, opened: bool):
            self.codec = codec
            self._opened = opened

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:
            released.append(self.codec)

    def _fake_fourcc(*chars):
        return "".join(chars)

    def _fake_writer_factory(path, fourcc, fps, frame_wh):
        attempted.append(fourcc)
        return _FakeWriter(fourcc, opened=(fourcc == "mp4v"))

    monkeypatch.setattr(video_demo.cv2, "VideoWriter_fourcc", _fake_fourcc)
    monkeypatch.setattr(video_demo.cv2, "VideoWriter", _fake_writer_factory)

    writer, chosen = video_demo._create_mp4_writer(tmp_path / "sub" / "out.mp4", 30.0, (640, 480))
    assert chosen == "mp4v"
    assert writer.codec == "mp4v"
    assert attempted == ["avc1", "H264", "mp4v"]
    assert released == ["avc1", "H264"]

    monkeypatch.setattr(
        video_demo.cv2,
        "VideoWriter",
        lambda *args: _FakeWriter(args[1], opened=False),
    )
    with pytest.raises(RuntimeError, match="Could not initialize MP4 writer"):
        video_demo._create_mp4_writer(tmp_path / "fail.mp4", 30.0, (640, 480))


def test_transcode_to_h264_if_needed_handles_h264_and_ffmpeg(tmp_path, monkeypatch):
    out_video = tmp_path / "clip.mp4"
    out_video.write_bytes(b"mp4v-bytes")

    monkeypatch.setattr(video_demo.shutil, "which", lambda _cmd: None)
    video_demo._transcode_to_h264_if_needed(out_video, "avc1")
    assert out_video.read_bytes() == b"mp4v-bytes"

    calls = []

    def _fake_run(cmd, check, capture_output):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"h264-bytes")

    monkeypatch.setattr(video_demo.shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_demo.subprocess, "run", _fake_run)

    video_demo._transcode_to_h264_if_needed(out_video, "avc1")
    assert calls == []

    video_demo._transcode_to_h264_if_needed(out_video, "mp4v")
    assert len(calls) == 1
    assert "libx264" in calls[0]
    assert "+faststart" in calls[0]
    assert out_video.read_bytes() == b"h264-bytes"
    assert not (tmp_path / "clip_h264.mp4").exists()