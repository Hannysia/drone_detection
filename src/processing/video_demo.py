from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable, Iterable

import cv2
from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd

from src import config
from src.config import YOLO_MODEL_STEM, get_quantized_model_paths
from src.processing.adaptive_roi import (
    AdaptiveROIConfig,
    AdaptiveROIRunner,
    DetectorLike,
    OpenVINOYOLODetector,
)

DEMO_MODES: tuple[str, ...] = ("roi_kalman", "roi_static", "full_1280", "resize_640")
DEMO_TRACKERS: tuple[str, ...] = ("bytetrack", "ocsort")

DEFAULT_DEMO_TRACKER_CONFIGS: dict[str, dict[str, float | int]] = {
    "bytetrack": {"track_thresh": 0.5, "match_thresh": 0.8, "track_buffer": 30},
    "ocsort": {"det_thresh": 0.5, "max_age": 30, "iou_threshold": 0.3},
}


@dataclass
class TrailState:
    """Maintains recent target centers and resets cleanly on track ID switch."""

    maxlen: int = 30
    active_track_id: int | None = None
    points: deque[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.points = deque(maxlen=max(1, int(self.maxlen)))

    def update(self, track_id: int | None, center_xy: tuple[int, int] | None) -> list[tuple[int, int]]:
        """Updates trail buffer with the latest center and returns ordered polyline points."""
        if track_id is not None and self.active_track_id is not None and track_id != self.active_track_id:
            self.points.clear()
        if track_id is not None:
            self.active_track_id = track_id
        if center_xy is not None:
            self.points.append((int(center_xy[0]), int(center_xy[1])))
        return list(self.points)


def _is_valid_openvino_dir(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    xml_files = sorted(model_dir.glob("*.xml"))
    if not xml_files:
        return False
    return xml_files[0].with_suffix(".bin").is_file()


def _validate_openvino_pair(model_1280: Path, model_640: Path) -> tuple[Path, Path]:
    if not _is_valid_openvino_dir(model_1280):
        raise FileNotFoundError(f"Invalid 1280 OpenVINO model directory: {model_1280}")
    if not _is_valid_openvino_dir(model_640):
        raise FileNotFoundError(f"Invalid 640 OpenVINO model directory: {model_640}")
    return model_1280, model_640


def resolve_demo_openvino_models(
    *,
    model_1280_dir: Path | None = None,
    model_640_dir: Path | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    model_stem: str = YOLO_MODEL_STEM,
) -> tuple[Path, Path]:
    """Resolves 1280 and 640 OpenVINO INT8 directories locally if present or via HF Hub."""
    if model_1280_dir is not None and model_640_dir is not None:
        return _validate_openvino_pair(Path(model_1280_dir), Path(model_640_dir))

    if os.getenv("LOCAL_BASE_DIR", "").strip():
        local_qdir = config.QUANTIZATION_DIR
        local_1280 = get_quantized_model_paths(local_qdir, 1280, model_stem=model_stem)["openvino_int8"]
        local_640 = get_quantized_model_paths(local_qdir, 640, model_stem=model_stem)["openvino_int8"]
        if _is_valid_openvino_dir(local_1280) and _is_valid_openvino_dir(local_640):
            return local_1280, local_640

    target_repo = (repo_id if repo_id is not None else os.getenv("REPO_ID") or config.REPO_ID or "").strip()
    if not target_repo:
        raise ValueError("REPO_ID is required to download quantized OpenVINO models from Hugging Face")
    target_token = token if token is not None else (os.getenv("HF_TOKEN") or config.HF_TOKEN)

    rel_1280 = f"models/quantization/1280/{model_stem}_int8_1280_openvino_model"
    rel_640 = f"models/quantization/640/{model_stem}_int8_640_openvino_model"
    download_kwargs: dict[str, object] = {
        "repo_id": target_repo,
        "repo_type": "dataset",
        "allow_patterns": [f"{rel_1280}/**", f"{rel_640}/**"],
    }
    if target_token:
        download_kwargs["token"] = target_token

    snapshot_root = Path(snapshot_download(**download_kwargs))
    return _validate_openvino_pair(snapshot_root / rel_1280, snapshot_root / rel_640)


def build_demo_detector_pair(
    *,
    model_1280_dir: Path | None = None,
    model_640_dir: Path | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    det_conf: float = 0.10,
) -> tuple[OpenVINOYOLODetector, OpenVINOYOLODetector]:
    """Loads and compiles the 1280 Search and 640 Track OpenVINO INT8 detectors once."""
    resolved_1280, resolved_640 = resolve_demo_openvino_models(
        model_1280_dir=model_1280_dir,
        model_640_dir=model_640_dir,
        repo_id=repo_id,
        token=token,
    )
    return (
        OpenVINOYOLODetector(resolved_1280, imgsz=1280, conf_thres=float(det_conf)),
        OpenVINOYOLODetector(resolved_640, imgsz=640, conf_thres=float(det_conf)),
    )


def load_demo_defaults(metadata_path: Path | None = None) -> dict[str, object]:
    """Loads calibrated tracker and ROI hysteresis defaults with fallback values."""
    defaults: dict[str, object] = {
        "tracker_configs": {k: dict(v) for k, v in DEFAULT_DEMO_TRACKER_CONFIGS.items()},
        "max_lost_frames": 4,
        "det_conf": 0.10,
        "seed_conf": 0.25,
        "roi_crop_size": 640,
    }
    if metadata_path is None or not Path(metadata_path).is_file():
        return defaults

    raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if isinstance(raw.get("tracker_configs"), dict):
        merged = dict(defaults["tracker_configs"])
        for name, cfg in raw["tracker_configs"].items():
            if isinstance(cfg, dict):
                merged[str(name)] = dict(cfg)
        defaults["tracker_configs"] = merged
    for key in ("max_lost_frames", "roi_crop_size"):
        if key in raw:
            defaults[key] = int(raw[key])
    for key in ("det_conf", "seed_conf"):
        if key in raw:
            defaults[key] = float(raw[key])
    return defaults


def _select_display_track_point(
    tracks: list[list[float]],
    target_track_id: int | None,
) -> tuple[int | None, tuple[int, int] | None]:
    if not tracks:
        return None, None
    chosen = None
    if target_track_id is not None:
        for row in tracks:
            if int(round(float(row[4]))) == int(target_track_id):
                chosen = row
                break
    if chosen is None:
        chosen = max(tracks, key=lambda r: float(r[5]))
    cx = int(round(0.5 * (float(chosen[0]) + float(chosen[2]))))
    cy = int(round(0.5 * (float(chosen[1]) + float(chosen[3]))))
    return int(round(float(chosen[4]))), (cx, cy)


def _draw_dashed_rect(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 12,
) -> None:
    x0, y0 = pt1
    x1, y1 = pt2
    for start, end in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        steps = max(1, int(length // (2 * dash_length)))
        for idx in range(steps + 1):
            t_start = min(1.0, (idx * 2 * dash_length) / length)
            t_end = min(1.0, ((idx * 2 + 1) * dash_length) / length)
            if t_end <= t_start:
                continue
            p_a = (int(round(start[0] + dx * t_start)), int(round(start[1] + dy * t_start)))
            p_b = (int(round(start[0] + dx * t_end)), int(round(start[1] + dy * t_end)))
            cv2.line(image, p_a, p_b, color, thickness, lineType=cv2.LINE_AA)


def render_demo_frame(
    bgr_frame: np.ndarray,
    record: dict[str, object],
    *,
    mode: str,
    trail_state: TrailState | None = None,
    draw_roi_box: bool = True,
    draw_trail: bool = True,
    draw_hud: bool = True,
) -> np.ndarray:
    """Renders tracked bboxes, optional trajectory trail, 640x640 ROI crop window, and HUD."""
    canvas = bgr_frame.copy()
    frame_h, frame_w = canvas.shape[:2]
    scale = max(0.55, min(frame_w, frame_h) / 720.0)
    box_thick = max(2, int(round(2.5 * scale)))
    font_scale = 0.65 * scale

    crop_box = record.get("crop_box")
    if draw_roi_box and crop_box is not None:
        x0, y0, x1, y1 = (int(v) for v in crop_box)
        _draw_dashed_rect(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 215, 255), box_thick)
        cv2.putText(
            canvas,
            f"ROI {x1 - x0}x{y1 - y0}",
            (x0 + 6, max(y0 + int(22 * scale), int(22 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            (0, 215, 255),
            max(1, box_thick - 1),
            cv2.LINE_AA,
        )

    tracks = list(record.get("tracks") or [])
    detections = list(record.get("detections") or [])
    target_id = record.get("target_track_id")

    if draw_trail and trail_state is not None:
        active_id, center_xy = _select_display_track_point(tracks, target_id)
        pts = trail_state.update(active_id, center_xy)
        n_pts = len(pts)
        for idx in range(1, n_pts):
            ratio = idx / max(1, n_pts - 1)
            color = (0, int(120 + 135 * ratio), int(255 * (1.0 - 0.5 * ratio)))
            cv2.line(canvas, pts[idx - 1], pts[idx], color, max(1, box_thick), lineType=cv2.LINE_AA)

    if tracks:
        for row in tracks:
            x1, y1, x2, y2 = (int(round(float(v))) for v in row[:4])
            tid = int(round(float(row[4])))
            conf = float(row[5])
            is_primary = target_id is None or tid == int(target_id)
            color = (60, 230, 80) if is_primary else (0, 165, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thick)
            cv2.putText(
                canvas,
                f"ID {tid} {conf:.2f}",
                (x1, max(y1 - 6, int(18 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, box_thick - 1),
                cv2.LINE_AA,
            )
    else:
        for det in detections:
            x1, y1, x2, y2 = (int(round(float(v))) for v in det[:4])
            conf = float(det[4])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), max(1, box_thick - 1))
            cv2.putText(
                canvas,
                f"det {conf:.2f}",
                (x1, max(y1 - 6, int(18 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * scale,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    if draw_hud:
        mode_used = str(record.get("mode_used", "search_1280"))
        latency_ms = float(record.get("latency_ms", 0.0))
        inst_fps = (1000.0 / latency_ms) if latency_ms > 0.0 else 0.0
        lost_frames = int(record.get("lost_frames", 0))
        is_track = mode_used == "track_640"
        badge_color = (60, 230, 80) if is_track else (0, 165, 255)
        mode_label = "TRACK 640" if is_track else "SEARCH 1280"
        line1 = f"{mode_label} | {inst_fps:.1f} FPS ({latency_ms:.1f} ms)"
        line2 = f"Pipeline: {mode} | Lost: {lost_frames} | Frame: {int(record.get('mot_frame', 0))}"

        pad = int(10 * scale)
        banner_w = min(frame_w - 2 * pad, int(460 * scale))
        banner_h = int(64 * scale)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + banner_w, pad + banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0, canvas)
        cv2.putText(
            canvas,
            line1,
            (pad + int(10 * scale), pad + int(25 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            badge_color,
            max(1, box_thick - 1),
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line2,
            (pad + int(10 * scale), pad + int(50 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _create_mp4_writer(output_path: Path, fps: float, frame_wh: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            frame_wh,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not initialize MP4 writer for {output_path}")


def _iter_video_frames(cap: cv2.VideoCapture, *, max_frames: int | None) -> Iterable[dict[str, object]]:
    idx = 0
    while max_frames is None or idx < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        yield {"mot_frame": idx, "dt_frames": 1.0, "bgr_image": frame}


def process_video_file(
    video_path: Path,
    *,
    output_video_path: Path | None = None,
    mode: str = "roi_kalman",
    tracker_name: str = "bytetrack",
    tracker_config: dict[str, float | int] | None = None,
    max_lost_frames: int = 4,
    search_conf: float = 0.10,
    track_conf: float = 0.10,
    seed_conf: float = 0.25,
    roi_crop_size: int = 640,
    draw_roi_box: bool = True,
    draw_trail: bool = True,
    draw_hud: bool = True,
    trail_length: int = 30,
    max_frames: int | None = None,
    search_detector: DetectorLike | None = None,
    track_detector: DetectorLike | None = None,
    model_1280_dir: Path | None = None,
    model_640_dir: Path | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Runs Adaptive ROI tracking and renders an annotated MP4 with per-frame latency telemetry."""
    if mode not in DEMO_MODES:
        raise ValueError(f"Unsupported demo mode {mode!r}; expected one of {DEMO_MODES}")
    if tracker_name.lower() not in DEMO_TRACKERS:
        raise ValueError(f"Unsupported tracker {tracker_name!r}; expected one of {DEMO_TRACKERS}")
    if max_lost_frames < 0:
        raise ValueError(f"max_lost_frames must be non-negative, got {max_lost_frames}")
    if roi_crop_size <= 0:
        raise ValueError(f"roi_crop_size must be positive, got {roi_crop_size}")
    if not (0.0 < search_conf <= 1.0 and 0.0 < track_conf <= 1.0 and 0.0 < seed_conf <= 1.0):
        raise ValueError("Confidence thresholds must lie in (0, 1]")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive when set, got {max_frames}")

    source_path = Path(video_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    needs_search = mode in ("full_1280", "roi_static", "roi_kalman")
    needs_track = mode in ("resize_640", "roi_static", "roi_kalman")
    if (needs_search and search_detector is None) or (needs_track and track_detector is None):
        compiled_1280, compiled_640 = build_demo_detector_pair(
            model_1280_dir=model_1280_dir,
            model_640_dir=model_640_dir,
            repo_id=repo_id,
            token=token,
            det_conf=min(search_conf, track_conf),
        )
        if search_detector is None:
            search_detector = compiled_1280
        if track_detector is None:
            track_detector = compiled_640

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not decode video stream: {source_path}")

    writer: cv2.VideoWriter | None = None
    try:
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
        video_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0.0 else 30.0
        estimated_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None and estimated_total > 0:
            estimated_total = min(estimated_total, int(max_frames))

        if output_video_path is None:
            out_path = Path(mkdtemp(prefix="drone_demo_")) / f"{source_path.stem}_{mode}.mp4"
        else:
            out_path = Path(output_video_path)

        resolved_tracker_cfg = (
            dict(tracker_config)
            if tracker_config is not None
            else dict(DEFAULT_DEMO_TRACKER_CONFIGS[tracker_name.lower()])
        )
        runner = AdaptiveROIRunner(
            search_detector=search_detector,
            track_detector=track_detector,
            config=AdaptiveROIConfig(
                mode=mode,
                tracker_name=tracker_name.lower(),
                tracker_config=resolved_tracker_cfg,
                fps=max(1, int(round(video_fps))),
                roi_crop_size=int(roi_crop_size),
                search_conf=float(search_conf),
                track_conf=float(track_conf),
                seed_conf=float(seed_conf),
                max_lost_frames=int(max_lost_frames),
            ),
        )

        trail_state = TrailState(maxlen=trail_length)
        per_frame_rows: list[dict[str, object]] = []
        frame_wh: tuple[int, int] | None = None

        def _on_frame(bgr_frame: np.ndarray, record: dict[str, object]) -> None:
            nonlocal writer, frame_wh
            if writer is None:
                frame_wh = (int(bgr_frame.shape[1]), int(bgr_frame.shape[0]))
                writer = _create_mp4_writer(out_path, video_fps, frame_wh)
            rendered = render_demo_frame(
                bgr_frame,
                record,
                mode=mode,
                trail_state=trail_state,
                draw_roi_box=draw_roi_box,
                draw_trail=draw_trail,
                draw_hud=draw_hud,
            )
            writer.write(rendered)

            latency_ms = float(record["latency_ms"])
            per_frame_rows.append({
                "frame_idx": len(per_frame_rows) + 1,
                "mot_frame": int(record["mot_frame"]),
                "mode_used": str(record["mode_used"]),
                "latency_ms": latency_ms,
                "inst_fps": (1000.0 / latency_ms) if latency_ms > 0.0 else 0.0,
                "num_tracks": len(record.get("tracks") or []),
                "target_track_id": record.get("target_track_id"),
                "lost_frames": int(record.get("lost_frames", 0)),
                "has_roi_crop": record.get("crop_box") is not None,
            })
            if progress_callback is not None:
                progress_callback(len(per_frame_rows), max(estimated_total, len(per_frame_rows)))

        seq_summary = runner.run_sequence(
            _iter_video_frames(cap, max_frames=max_frames),
            on_frame=_on_frame,
        )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Output video was not written: {out_path}")

    summary = {
        "mode": mode,
        "tracker_name": tracker_name.lower(),
        "video_fps": float(video_fps),
        "width": frame_wh[0] if frame_wh is not None else 0,
        "height": frame_wh[1] if frame_wh is not None else 0,
        "total_frames": int(seq_summary["total_frames"]),
        "search_frames": int(seq_summary["search_frames"]),
        "track_frames": int(seq_summary["track_frames"]),
        "pct_track_frames": float(seq_summary["pct_track_frames"]),
        "search_to_track_transitions": int(seq_summary["search_to_track_transitions"]),
        "track_to_search_transitions": int(seq_summary["track_to_search_transitions"]),
        "search_transitions": int(seq_summary["search_transitions"]),
        "e2e_mean_ms": float(seq_summary["e2e_mean_ms"]),
        "e2e_p50_ms": float(seq_summary["e2e_p50_ms"]),
        "e2e_p95_ms": float(seq_summary["e2e_p95_ms"]),
        "e2e_fps": float(seq_summary["e2e_fps"]),
    }
    return {
        "output_video_path": out_path,
        "summary": summary,
        "per_frame_df": pd.DataFrame(per_frame_rows),
    }