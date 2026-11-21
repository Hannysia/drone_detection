from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Literal, Protocol

import cv2
import numpy as np

from src.processing.diagnostics import letterbox_image
from src.processing.tracking import get_configured_tracker


class DetectorLike(Protocol):
    def predict(self, bgr_image: np.ndarray, *, conf_thres: float | None = None) -> np.ndarray:
        ...


def _build_openvino_runner(model_dir: Path) -> Callable[[np.ndarray], np.ndarray]:
    import openvino as ov

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"OpenVINO model directory not found: {model_dir}")
    xml_files = sorted(model_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml model file found in {model_dir}")
    xml_path = xml_files[0]
    bin_path = xml_path.with_suffix(".bin")
    if not bin_path.is_file():
        raise FileNotFoundError(f"Missing OpenVINO weights file: {bin_path}")

    core = ov.Core()
    ov_model = core.read_model(model=str(xml_path), weights=str(bin_path))
    compiled = core.compile_model(
        ov_model,
        device_name="CPU",
        config={"PERFORMANCE_HINT": "LATENCY"},
    )
    output_port = compiled.output(0)

    def _run(tensor: np.ndarray) -> np.ndarray:
        result = compiled(tensor)
        return np.asarray(result[output_port], dtype=np.float32)

    return _run


def decode_end2end_predictions(
    raw_output: np.ndarray | dict,
    orig_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    *,
    conf_thres: float = 0.25,
    target_classes: tuple[int, ...] | None = (0,),
    max_det: int = 300,
) -> np.ndarray:
    """Decodes end2end YOLO predictions (N, 6) from letterboxed coordinates to orig_shape."""
    if isinstance(raw_output, dict):
        arr = np.asarray(next(iter(raw_output.values())), dtype=np.float32)
    else:
        arr = np.asarray(raw_output, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return np.empty((0, 6), dtype=np.float32)
    if arr.shape[1] != 6 and arr.shape[0] == 6:
        arr = arr.T
    if arr.shape[0] == 0 or arr.shape[1] != 6:
        return np.empty((0, 6), dtype=np.float32)

    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]
    if arr.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)

    mask = arr[:, 4] >= float(conf_thres)
    if target_classes is not None:
        cls_ids = np.rint(arr[:, 5]).astype(np.int64)
        mask &= np.isin(cls_ids, target_classes)
    filtered = arr[mask].copy()
    if filtered.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)

    order = np.argsort(-filtered[:, 4], kind="stable")
    filtered = filtered[order[:max_det]]

    orig_h, orig_w = orig_shape
    top, bottom, left, right = content_box
    scale_x = max(right - left, 1) / float(orig_w)
    scale_y = max(bottom - top, 1) / float(orig_h)

    filtered[:, [0, 2]] = (filtered[:, [0, 2]] - float(left)) / scale_x
    filtered[:, [1, 3]] = (filtered[:, [1, 3]] - float(top)) / scale_y
    filtered[:, [0, 2]] = np.clip(filtered[:, [0, 2]], 0.0, float(orig_w))
    filtered[:, [1, 3]] = np.clip(filtered[:, [1, 3]], 0.0, float(orig_h))

    valid = (filtered[:, 2] > filtered[:, 0]) & (filtered[:, 3] > filtered[:, 1])
    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)
    return np.ascontiguousarray(filtered[valid], dtype=np.float32)


class OpenVINOYOLODetector:
    """Direct OpenVINO CPU runner and end2end decoder without YOLO.predict wrapper overhead."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        imgsz: int,
        conf_thres: float = 0.25,
        target_classes: tuple[int, ...] | None = (0,),
        max_det: int = 300,
        runner: Callable[[np.ndarray], np.ndarray | dict] | None = None,
    ) -> None:
        if imgsz <= 0 or imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a positive multiple of 32, got {imgsz}")
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.target_classes = target_classes
        self.max_det = int(max_det)
        if runner is not None:
            self._runner = runner
        else:
            if model_dir is None:
                raise ValueError("Provide either model_dir or a custom runner")
            self._runner = _build_openvino_runner(model_dir)

    def predict(self, bgr_image: np.ndarray, *, conf_thres: float | None = None) -> np.ndarray:
        """Runs preprocessing, compiled inference, and coordinate rescaling to input frame shape."""
        if bgr_image is None or bgr_image.ndim != 3 or bgr_image.shape[0] == 0 or bgr_image.shape[1] == 0:
            raise ValueError("bgr_image must be a non-empty HWC image")
        orig_h, orig_w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        padded, content_box = letterbox_image(rgb, imgsz=self.imgsz)
        chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32) * (1.0 / 255.0)
        raw_output = self._runner(chw[None, ...])
        threshold = self.conf_thres if conf_thres is None else float(conf_thres)
        return decode_end2end_predictions(
            raw_output,
            (orig_h, orig_w),
            content_box,
            conf_thres=threshold,
            target_classes=self.target_classes,
            max_det=self.max_det,
        )


def compute_roi_crop_window(
    center_xy: tuple[float, float],
    frame_wh: tuple[int, int],
    crop_size: int = 640,
) -> tuple[int, int, int, int]:
    """Computes an integer [x0, y0, x1, y1] ROI crop clamped within frame boundaries."""
    frame_w, frame_h = int(frame_wh[0]), int(frame_wh[1])
    if frame_w <= 0 or frame_h <= 0 or crop_size <= 0:
        raise ValueError(f"Invalid frame_wh={frame_wh} or crop_size={crop_size}")
    cw = min(int(crop_size), frame_w)
    ch = min(int(crop_size), frame_h)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x0 = int(round(cx - cw / 2.0))
    y0 = int(round(cy - ch / 2.0))
    x0 = max(0, min(x0, frame_w - cw))
    y0 = max(0, min(y0, frame_h - ch))
    return x0, y0, x0 + cw, y0 + ch


def map_crop_detections_to_global(
    dets_crop: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Shifts crop-local (N, 6) detections back into full-frame global coordinates."""
    arr = np.asarray(dets_crop, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    arr = arr.reshape(-1, 6).copy()
    x0, y0 = float(crop_box[0]), float(crop_box[1])
    arr[:, [0, 2]] += x0
    arr[:, [1, 3]] += y0
    return arr


def _clamp_xy(
    x: float,
    y: float,
    frame_wh: tuple[int, int] | None,
) -> tuple[float, float]:
    if frame_wh is None:
        return float(x), float(y)
    return (
        float(np.clip(x, 0.0, float(frame_wh[0]))),
        float(np.clip(y, 0.0, float(frame_wh[1]))),
    )


class ROICenterForecaster:
    """Predicts next ROI crop center in static or gap-safe 2D Kalman mode."""

    def __init__(
        self,
        mode: Literal["static", "kalman"] = "kalman",
        *,
        max_dt_steps: float = 5.0,
        lost_velocity_decay: float = 0.85,
        gap_velocity_decay: float = 0.5,
        max_extrapolation_px: float = 220.0,
        process_pos_var: float = 4.0,
        process_vel_var: float = 16.0,
        meas_pos_var: float = 25.0,
    ) -> None:
        if mode not in ("static", "kalman"):
            raise ValueError(f"Unsupported forecaster mode: {mode!r}")
        self.mode = mode
        self.max_dt_steps = float(max_dt_steps)
        self.lost_velocity_decay = float(lost_velocity_decay)
        self.gap_velocity_decay = float(gap_velocity_decay)
        self.max_extrapolation_px = float(max_extrapolation_px)
        self._q_pos = float(process_pos_var)
        self._q_vel = float(process_vel_var)
        self._h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._r = np.diag([float(meas_pos_var), float(meas_pos_var)])
        self._state: np.ndarray | None = None
        self._cov: np.ndarray | None = None

    def reset(self) -> None:
        """Clears internal target motion state."""
        self._state = None
        self._cov = None

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    def _effective_dt(self, dt_frames: float) -> tuple[float, bool]:
        raw_dt = max(float(dt_frames), 1.0)
        is_large_gap = raw_dt > self.max_dt_steps
        return min(raw_dt, self.max_dt_steps), is_large_gap

    def predict_center(
        self,
        dt_next: float = 1.0,
        *,
        frame_wh: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        """Returns predicted ROI crop center for the upcoming frame."""
        if self._state is None:
            raise RuntimeError("Forecaster has not been initialized with an observation")
        if self.mode == "static":
            return _clamp_xy(self._state[0], self._state[1], frame_wh)

        eff_dt, is_large_gap = self._effective_dt(dt_next)
        vx, vy = float(self._state[2]), float(self._state[3])
        if is_large_gap:
            gap_steps = min(float(dt_next) - 1.0, 10.0)
            scale = self.gap_velocity_decay ** gap_steps
            vx *= scale
            vy *= scale
        dx = float(np.clip(vx * eff_dt, -self.max_extrapolation_px, self.max_extrapolation_px))
        dy = float(np.clip(vy * eff_dt, -self.max_extrapolation_px, self.max_extrapolation_px))
        return _clamp_xy(self._state[0] + dx, self._state[1] + dy, frame_wh)

    def update(
        self,
        center_xy: tuple[float, float] | None,
        *,
        dt_frames: float = 1.0,
        frame_wh: tuple[int, int] | None = None,
    ) -> tuple[float, float] | None:
        """Advances state by dt_frames and optionally assimilates an observed target center."""
        if center_xy is not None:
            cx, cy = _clamp_xy(center_xy[0], center_xy[1], frame_wh)
            if self._state is None:
                self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
                self._cov = np.diag([25.0, 25.0, 100.0, 100.0])
                return cx, cy

            if self.mode == "static":
                self._state[0], self._state[1] = cx, cy
                return cx, cy

        if self._state is None:
            return None
        if self.mode == "static":
            return _clamp_xy(self._state[0], self._state[1], frame_wh)

        eff_dt, is_large_gap = self._effective_dt(dt_frames)
        if is_large_gap:
            gap_steps = min(float(dt_frames) - 1.0, 10.0)
            self._state[2:] *= self.gap_velocity_decay ** gap_steps
            self._cov[2, 2] = max(self._cov[2, 2], 100.0)
            self._cov[3, 3] = max(self._cov[3, 3], 100.0)

        f = np.array(
            [
                [1.0, 0.0, eff_dt, 0.0],
                [0.0, 1.0, 0.0, eff_dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = np.diag([self._q_pos * eff_dt, self._q_pos * eff_dt, self._q_vel * eff_dt, self._q_vel * eff_dt])
        self._state = f @ self._state
        self._cov = f @ self._cov @ f.T + q

        if center_xy is None:
            self._state[2:] *= self.lost_velocity_decay
            clamped_x, clamped_y = _clamp_xy(self._state[0], self._state[1], frame_wh)
            self._state[0], self._state[1] = clamped_x, clamped_y
            return clamped_x, clamped_y

        z = np.array([cx, cy], dtype=np.float64)
        innov = z - self._h @ self._state
        s = self._h @ self._cov @ self._h.T + self._r
        k = self._cov @ self._h.T @ np.linalg.inv(s)
        self._state = self._state + k @ innov
        i_kh = np.eye(4, dtype=np.float64) - k @ self._h
        self._cov = i_kh @ self._cov @ i_kh.T + k @ self._r @ k.T
        return _clamp_xy(self._state[0], self._state[1], frame_wh)


@dataclass(frozen=True)
class AdaptiveROIConfig:
    mode: Literal["full_1280", "resize_640", "roi_static", "roi_kalman"] = "roi_kalman"
    tracker_name: str = "bytetrack"
    tracker_config: dict = field(default_factory=dict)
    fps: int = 30
    search_imgsz: int = 1280
    track_imgsz: int = 640
    roi_crop_size: int = 640
    search_conf: float = 0.25
    track_conf: float = 0.15
    seed_conf: float = 0.25
    max_lost_frames: int = 4
    forecaster_kwargs: dict = field(default_factory=dict)


def _normalize_tracker_outputs(raw_targets: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_targets, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 7:
        raise ValueError(f"Expected tracker output with at least 7 columns, got shape {arr.shape}")
    return arr


def _select_target_observation(
    tracks: np.ndarray,
    global_dets: np.ndarray,
    *,
    target_track_id: int | None,
    allow_seed_det: bool,
    seed_conf: float,
    anchor_xy: tuple[float, float] | None,
) -> tuple[int | None, tuple[float, float] | None]:
    if tracks.shape[0] > 0:
        if target_track_id is not None:
            for row in tracks:
                if int(round(float(row[4]))) == target_track_id:
                    cx = 0.5 * (float(row[0]) + float(row[2]))
                    cy = 0.5 * (float(row[1]) + float(row[3]))
                    return target_track_id, (cx, cy)

        if anchor_xy is not None:
            ax, ay = anchor_xy
            best_row = min(
                tracks,
                key=lambda r: (0.5 * (float(r[0]) + float(r[2])) - ax) ** 2
                + (0.5 * (float(r[1]) + float(r[3])) - ay) ** 2,
            )
        else:
            best_row = tracks[int(np.argmax(tracks[:, 5]))]
        chosen_id = int(round(float(best_row[4])))
        cx = 0.5 * (float(best_row[0]) + float(best_row[2]))
        cy = 0.5 * (float(best_row[1]) + float(best_row[3]))
        return chosen_id, (cx, cy)

    if allow_seed_det and global_dets.shape[0] > 0:
        seed_dets = global_dets[global_dets[:, 4] >= seed_conf]
        if seed_dets.shape[0] > 0:
            if anchor_xy is not None:
                ax, ay = anchor_xy
                best_det = min(
                    seed_dets,
                    key=lambda r: (0.5 * (float(r[0]) + float(r[2])) - ax) ** 2
                    + (0.5 * (float(r[1]) + float(r[3])) - ay) ** 2,
                )
            else:
                best_det = seed_dets[int(np.argmax(seed_dets[:, 4]))]
            cx = 0.5 * (float(best_det[0]) + float(best_det[2]))
            cy = 0.5 * (float(best_det[1]) + float(best_det[3]))
            return None, (cx, cy)

    return target_track_id, None


def _read_sequence_frame(frame_spec: dict[str, object]) -> np.ndarray:
    if "bgr_image" in frame_spec and frame_spec["bgr_image"] is not None:
        return np.asarray(frame_spec["bgr_image"], dtype=np.uint8)
    image_path = Path(str(frame_spec["image_path"]))
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Unreadable sequence frame: {image_path}")
    return img


def _summarize_sequence_metrics(
    latencies_ms: list[float],
    search_frames: int,
    track_frames: int,
    search_to_track: int,
    track_to_search: int,
) -> dict[str, float | int]:
    total_frames = len(latencies_ms)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr)) if total_frames > 0 else 0.0
    return {
        "total_frames": total_frames,
        "search_frames": int(search_frames),
        "track_frames": int(track_frames),
        "pct_track_frames": (track_frames / total_frames * 100.0) if total_frames > 0 else 0.0,
        "search_to_track_transitions": int(search_to_track),
        "track_to_search_transitions": int(track_to_search),
        "search_transitions": int(search_to_track + track_to_search),
        "e2e_mean_ms": mean_ms,
        "e2e_p50_ms": float(np.percentile(arr, 50)) if total_frames > 0 else 0.0,
        "e2e_p95_ms": float(np.percentile(arr, 95)) if total_frames > 0 else 0.0,
        "e2e_fps": (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0,
    }


def _write_mot_results(output_txt_path: Path, mot_rows: list[list[float]]) -> None:
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    if mot_rows:
        np.savetxt(
            output_txt_path,
            np.asarray(mot_rows, dtype=np.float64),
            fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d",
        )
    else:
        output_txt_path.write_text("", encoding="utf-8")


class AdaptiveROIRunner:
    """Runs full_1280, resize_640, roi_static, or roi_kalman detection and tracking on video sequences."""

    def __init__(
        self,
        *,
        search_detector: DetectorLike | None = None,
        track_detector: DetectorLike | None = None,
        config: AdaptiveROIConfig | None = None,
    ) -> None:
        self.config = config or AdaptiveROIConfig()
        if self.config.mode in ("full_1280", "roi_static", "roi_kalman") and search_detector is None:
            raise ValueError(f"search_detector is required for mode={self.config.mode!r}")
        if self.config.mode in ("resize_640", "roi_static", "roi_kalman") and track_detector is None:
            raise ValueError(f"track_detector is required for mode={self.config.mode!r}")
        if self.config.max_lost_frames < 0:
            raise ValueError(f"max_lost_frames must be non-negative, got {self.config.max_lost_frames}")
        self.search_detector = search_detector
        self.track_detector = track_detector

    def _create_tracker(self):
        return get_configured_tracker(
            self.config.tracker_name,
            dict(self.config.tracker_config),
            fps=int(self.config.fps),
        )

    def _create_forecaster(self) -> ROICenterForecaster:
        mode: Literal["static", "kalman"] = "static" if self.config.mode == "roi_static" else "kalman"
        return ROICenterForecaster(mode=mode, **dict(self.config.forecaster_kwargs))

    def _detect_frame(
        self,
        bgr_frame: np.ndarray,
        *,
        use_track_roi: bool,
        use_full_track_model: bool,
        forecaster: ROICenterForecaster,
        dt_frames: float,
    ) -> tuple[np.ndarray, str, tuple[int, int, int, int] | None]:
        frame_h, frame_w = bgr_frame.shape[:2]
        if use_full_track_model:
            dets = self.track_detector.predict(bgr_frame, conf_thres=self.config.track_conf)
            return dets, "track_640", None
        if use_track_roi:
            anchor = forecaster.predict_center(dt_next=dt_frames, frame_wh=(frame_w, frame_h))
            crop_box = compute_roi_crop_window(
                anchor,
                (frame_w, frame_h),
                crop_size=self.config.roi_crop_size,
            )
            x0, y0, x1, y1 = crop_box
            crop_bgr = bgr_frame[y0:y1, x0:x1]
            crop_dets = self.track_detector.predict(crop_bgr, conf_thres=self.config.track_conf)
            return map_crop_detections_to_global(crop_dets, crop_box), "track_640", crop_box
        dets = self.search_detector.predict(bgr_frame, conf_thres=self.config.search_conf)
        return dets, "search_1280", None

    def run_sequence(
        self,
        frames: list[dict[str, object]],
        *,
        frame_loader: Callable[[dict[str, object]], np.ndarray] = _read_sequence_frame,
        output_txt_path: Path | None = None,
        return_frame_records: bool = False,
    ) -> dict[str, object]:
        """Processes one chronological frame sequence and returns MOT outputs plus E2E timing telemetry."""
        if not frames:
            raise ValueError("Sequence frames must not be empty")

        tracker = self._create_tracker()
        forecaster = self._create_forecaster()
        is_roi_mode = self.config.mode in ("roi_static", "roi_kalman")
        is_resize_640 = self.config.mode == "resize_640"

        state = "track" if is_resize_640 else "search"
        target_track_id: int | None = None
        lost_frames = 0
        search_frames = 0
        track_frames = 0
        search_to_track = 0
        track_to_search = 0
        latencies_ms: list[float] = []
        mot_rows: list[list[float]] = []
        frame_records: list[dict[str, object]] = []

        for idx, frame_spec in enumerate(frames, start=1):
            bgr_frame = frame_loader(frame_spec)
            frame_h, frame_w = bgr_frame.shape[:2]
            mot_frame = int(frame_spec.get("mot_frame", idx))
            dt_frames = float(frame_spec.get("dt_frames", 1.0))

            t0 = time.perf_counter()
            use_roi = is_roi_mode and state == "track" and forecaster.is_initialized
            global_dets, mode_used, crop_box = self._detect_frame(
                bgr_frame,
                use_track_roi=use_roi,
                use_full_track_model=is_resize_640,
                forecaster=forecaster,
                dt_frames=dt_frames,
            )
            tracker_in = (
                global_dets if global_dets.shape[0] > 0 else np.empty((0, 6), dtype=np.float32)
            )
            tracks = _normalize_tracker_outputs(tracker.update(tracker_in, bgr_frame))

            if is_roi_mode:
                anchor_xy = (
                    forecaster.predict_center(dt_next=dt_frames, frame_wh=(frame_w, frame_h))
                    if forecaster.is_initialized
                    else None
                )
                chosen_id, obs_center = _select_target_observation(
                    tracks,
                    global_dets,
                    target_track_id=target_track_id,
                    allow_seed_det=(mode_used == "search_1280" or target_track_id is None),
                    seed_conf=self.config.seed_conf,
                    anchor_xy=anchor_xy,
                )
                if obs_center is not None:
                    forecaster.update(obs_center, dt_frames=dt_frames, frame_wh=(frame_w, frame_h))
                    target_track_id = chosen_id
                    lost_frames = 0
                    if state != "track":
                        state = "track"
                        search_to_track += 1
                elif state == "track":
                    lost_frames += 1
                    forecaster.update(None, dt_frames=dt_frames, frame_wh=(frame_w, frame_h))
                    if target_track_id is None or lost_frames > self.config.max_lost_frames:
                        state = "search"
                        target_track_id = None
                        lost_frames = 0
                        forecaster.reset()
                        track_to_search += 1

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            if mode_used == "track_640":
                track_frames += 1
            else:
                search_frames += 1

            for row in tracks:
                x1, y1, x2, y2, tid, conf = (
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    int(round(float(row[4]))),
                    float(row[5]),
                )
                mot_rows.append([mot_frame, tid, x1, y1, x2 - x1, y2 - y1, conf, -1, -1, -1])

            if return_frame_records:
                frame_records.append({
                    "mot_frame": mot_frame,
                    "mode_used": mode_used,
                    "state_after": state,
                    "crop_box": crop_box,
                    "lost_frames": lost_frames,
                    "target_track_id": target_track_id,
                    "detections": global_dets.tolist(),
                    "tracks": tracks[:, :6].tolist() if tracks.shape[0] > 0 else [],
                    "latency_ms": elapsed_ms,
                })

        if output_txt_path is not None:
            _write_mot_results(output_txt_path, mot_rows)

        summary = _summarize_sequence_metrics(
            latencies_ms,
            search_frames,
            track_frames,
            search_to_track,
            track_to_search,
        )
        result: dict[str, object] = {
            **summary,
            "latencies_ms": latencies_ms,
            "mot_rows": mot_rows,
        }
        if return_frame_records:
            result["frame_records"] = frame_records
        return result


def run_adaptive_roi_on_sequences(
    sequences: dict[str, list[dict[str, object]]],
    *,
    search_detector: DetectorLike | None = None,
    track_detector: DetectorLike | None = None,
    config: AdaptiveROIConfig | None = None,
    output_dir: Path | None = None,
    frame_loader: Callable[[dict[str, object]], np.ndarray] = _read_sequence_frame,
) -> dict[str, object]:
    """Runs adaptive ROI pipeline across all video sequences and aggregates latency and mode metrics."""
    if not sequences:
        raise ValueError("sequences dictionary must not be empty")
    runner = AdaptiveROIRunner(
        search_detector=search_detector,
        track_detector=track_detector,
        config=config,
    )
    all_latencies: list[float] = []
    total_search_frames = 0
    total_track_frames = 0
    total_s2t = 0
    total_t2s = 0
    per_sequence: dict[str, dict[str, object]] = {}

    for seq_name in sorted(sequences):
        out_txt = (Path(output_dir) / f"{seq_name}.txt") if output_dir is not None else None
        seq_res = runner.run_sequence(
            sequences[seq_name],
            frame_loader=frame_loader,
            output_txt_path=out_txt,
        )
        all_latencies.extend(seq_res["latencies_ms"])
        total_search_frames += int(seq_res["search_frames"])
        total_track_frames += int(seq_res["track_frames"])
        total_s2t += int(seq_res["search_to_track_transitions"])
        total_t2s += int(seq_res["track_to_search_transitions"])
        per_sequence[seq_name] = {
            k: v for k, v in seq_res.items() if k not in ("latencies_ms", "mot_rows")
        }

    overall = _summarize_sequence_metrics(
        all_latencies,
        total_search_frames,
        total_track_frames,
        total_s2t,
        total_t2s,
    )
    return {
        **overall,
        "num_sequences": len(per_sequence),
        "per_sequence": per_sequence,
    }