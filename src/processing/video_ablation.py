from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd

from src import config
from src.config import (
    MANIFESTS_DIR,
    VIDEO_EXTENSIONS,
    get_quantized_model_paths,
)
from src.processing.adaptive_roi import (
    AdaptiveROIConfig,
    DetectorLike,
    OpenVINOYOLODetector,
    _read_sequence_frame,
    run_adaptive_roi_on_sequences,
)
from src.processing.evaluation import (
    _greedy_match_predictions,
    parse_trackeval_summary,
    run_trackeval,
)
from src.processing.manifests import load_baseline_video_sequences
from src.processing.trackeval_dataset import compile_kaggle_mot_dataset
from src.processing.tracking import run_tracking_pipeline

log = logging.getLogger(__name__)

ABLATION_MODES: tuple[str, ...] = ("full_1280", "resize_640", "roi_static", "roi_kalman")
ABLATION_TRACKERS: tuple[str, ...] = ("bytetrack", "ocsort")
DEFAULT_VAL_LOST_FRAMES: tuple[int, ...] = (3, 4, 5)

DEFAULT_BYTETRACK_GRID: dict[str, list[float | int]] = {
    "track_thresh": [0.3, 0.4, 0.5, 0.6, 0.7],
    "track_buffer": [15, 30, 45, 60],
    "match_thresh": [0.7, 0.8, 0.9],
}

DEFAULT_OCSORT_GRID: dict[str, list[float | int]] = {
    "det_thresh": [0.3, 0.4, 0.5, 0.6, 0.7],
    "max_age": [15, 30, 45, 60],
    "iou_threshold": [0.2, 0.3, 0.4, 0.5],
}


def _grid_combinations(param_grid: dict[str, list[float | int]]) -> list[dict[str, float | int]]:
    keys = list(param_grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(param_grid[k] for k in keys))]


def _read_seqmap_names(seqmaps_dir: Path, split: str) -> list[str]:
    seqmap_file = Path(seqmaps_dir) / f"drone-{split}.txt"
    if not seqmap_file.is_file():
        raise FileNotFoundError(f"Missing TrackEval seqmap: {seqmap_file}")
    lines = [line.strip() for line in seqmap_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[1:] if lines and lines[0].lower() == "name" else lines


def load_curated_ablation_sequences(
    dataset_root: Path,
    seqmaps_dir: Path,
    *,
    split: str,
    manifest_path: Path = MANIFESTS_DIR / "baseline" / "v1" / "main.csv",
    source_path: Path | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Loads baseline video sequences filtered and ordered by the curated TrackEval seqmap."""
    all_sequences = load_baseline_video_sequences(
        Path(dataset_root),
        split=split,
        manifest_path=Path(manifest_path),
        source_path=Path(source_path) if source_path is not None else None,
        allowed_splits=("val", "test"),
    )
    seqmap_names = _read_seqmap_names(Path(seqmaps_dir), split)
    missing = [name for name in seqmap_names if name not in all_sequences]
    if missing:
        raise ValueError(f"Seqmap sequences missing from baseline dataset ({split}): {missing}")
    if not seqmap_names:
        raise ValueError(f"No sequences found in seqmap for split={split!r}")
    return {name: all_sequences[name] for name in seqmap_names}


def _resolve_curated_hf_source(
    source_path: Path | None,
    *,
    repo_id: str | None,
    token: str | None,
    annotation_version: str,
) -> tuple[str | None, str, str | None, str, str | None]:
    source_meta: dict[str, object] = {}
    if source_path is not None and Path(source_path).is_file():
        source_meta = json.loads(Path(source_path).read_text(encoding="utf-8"))

    target_repo = (
        repo_id
        if repo_id is not None
        else source_meta.get("repo_id") or getattr(config, "REPO_ID", None)
    )
    repo_str = str(target_repo).strip() if target_repo else None
    repo_type = str(source_meta.get("repo_type") or "dataset").strip() or "dataset"
    raw_rev = source_meta.get("revision")
    revision = str(raw_rev).strip() if raw_rev else None
    version = str(source_meta.get("annotation_version") or annotation_version).strip() or annotation_version
    target_token = token if token is not None else getattr(config, "HF_TOKEN", None)
    token_str = str(target_token).strip() if target_token else None
    return repo_str, repo_type, revision, version, token_str


def ensure_curated_annotations(
    hf_dir: Path,
    *,
    splits: tuple[str, ...] = ("val", "test"),
    annotation_version: str = "v1",
    source_path: Path | None = None,
    repo_id: str | None = None,
    token: str | None = None,
) -> str:
    """Ensures curated annotation archives exist locally, downloading missing splits from HF Hub."""
    hf_root = Path(hf_dir)
    repo_str, repo_type, revision, version, token_str = _resolve_curated_hf_source(
        source_path,
        repo_id=repo_id,
        token=token,
        annotation_version=annotation_version,
    )
    curated_root = hf_root / "labels_curated" / version
    missing_patterns = [
        f"labels_curated/{version}/{split}/**"
        for split in splits
        if not (curated_root / split).is_dir() or not any((curated_root / split).rglob("*.zip"))
    ]
    if not missing_patterns or not repo_str:
        return version

    hf_root.mkdir(parents=True, exist_ok=True)
    download_kwargs: dict[str, object] = {
        "repo_id": repo_str,
        "repo_type": repo_type,
        "local_dir": str(hf_root),
        "allow_patterns": missing_patterns,
    }
    if revision:
        download_kwargs["revision"] = revision
    if token_str:
        download_kwargs["token"] = token_str

    log.info("Downloading curated annotations (%s) from %s -> %s", ", ".join(splits), repo_str, hf_root)
    snapshot_download(**download_kwargs)
    return version


def prepare_curated_mot_ground_truth(
    hf_dir: Path,
    dataset_root: Path,
    mot_dataset_dir: Path,
    seqmaps_dir: Path,
    *,
    splits: tuple[str, ...] = ("val", "test"),
    manifest_path: Path = MANIFESTS_DIR / "baseline" / "v1" / "main.csv",
    source_path: Path | None = None,
    fps: float = 30.0,
    repo_id: str | None = None,
    token: str | None = None,
) -> dict[str, list[str]]:
    """Builds curated TrackEval ground truth aligned to baseline/v1 frames."""
    annotation_version = ensure_curated_annotations(
        Path(hf_dir),
        splits=splits,
        source_path=Path(source_path) if source_path is not None else None,
        repo_id=repo_id,
        token=token,
    )
    return compile_kaggle_mot_dataset(
        Path(hf_dir),
        Path(mot_dataset_dir),
        Path(seqmaps_dir),
        tuple(VIDEO_EXTENSIONS),
        annotation_version=annotation_version,
        splits=splits,
        baseline_dataset_root=Path(dataset_root),
        manifest_path=Path(manifest_path),
        source_path=Path(source_path) if source_path is not None else None,
        default_fps=float(fps),
    )


def collect_openvino_detections_cache(
    sequences: dict[str, list[dict[str, object]]],
    detector: DetectorLike,
    cache_dir: Path,
    *,
    conf_thres: float = 0.10,
    resume: bool = False,
    frame_loader: Callable[[dict[str, object]], np.ndarray] = _read_sequence_frame,
) -> Path:
    """Runs detector once per sequence and caches MOT detection files for tracker grid search."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in sorted(sequences):
        out_file = cache_dir / f"{seq_name}.txt"
        if resume and out_file.is_file():
            continue
        rows: list[list[float]] = []
        for idx, frame_spec in enumerate(sequences[seq_name], start=1):
            bgr = frame_loader(frame_spec)
            mot_frame = int(frame_spec.get("mot_frame", idx))
            preds = detector.predict(bgr, conf_thres=float(conf_thres))
            for det in preds:
                x1, y1, x2, y2, score, cls_id = det[:6]
                rows.append([mot_frame, -1, float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(score), int(round(float(cls_id))), -1])
        if rows:
            np.savetxt(
                out_file,
                np.asarray(rows, dtype=np.float64),
                fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d",
            )
        else:
            out_file.write_text("", encoding="utf-8")
    return cache_dir


def _load_mot_boxes_by_frame(
    file_path: Path,
    *,
    is_gt: bool,
) -> dict[int, list[list[float]]]:
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return {}
    raw = np.loadtxt(file_path, delimiter=",", ndmin=2)
    by_frame: dict[int, list[list[float]]] = {}
    for row in raw:
        if len(row) < 6:
            continue
        frame_id = int(round(float(row[0])))
        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        if w <= 0.0 or h <= 0.0:
            continue
        if is_gt and len(row) >= 7 and int(round(float(row[6]))) == 0:
            continue
        x2, y2 = x + w, y + h
        if is_gt:
            by_frame.setdefault(frame_id, []).append([x, y, x2, y2])
        else:
            score = float(row[6]) if len(row) >= 7 else 1.0
            by_frame.setdefault(frame_id, []).append([x, y, x2, y2, score])
    return by_frame


def compute_track_small_recall(
    gt_split_dir: Path,
    tracker_data_dir: Path,
    sequence_names: list[str],
    *,
    iou_threshold: float = 0.5,
    small_area_threshold: float = 1024.0,
) -> dict[str, float | int]:
    """Computes COCO small-target (<1024 px^2) and overall recall/precision on tracking outputs."""
    gt_split_dir = Path(gt_split_dir)
    tracker_data_dir = Path(tracker_data_dir)
    gt_total = 0
    tp_total = 0
    fp_total = 0
    gt_small = 0
    tp_small = 0

    for seq_name in sequence_names:
        gt_by_frame = _load_mot_boxes_by_frame(gt_split_dir / seq_name / "gt" / "gt.txt", is_gt=True)
        pred_by_frame = _load_mot_boxes_by_frame(tracker_data_dir / f"{seq_name}.txt", is_gt=False)
        all_frames = set(gt_by_frame.keys()) | set(pred_by_frame.keys())
        for frame_id in all_frames:
            gts = gt_by_frame.get(frame_id, [])
            preds = pred_by_frame.get(frame_id, [])
            tp_gt_pairs, fp_indices, _, _ = _greedy_match_predictions(
                gts,
                preds,
                iou_threshold=iou_threshold,
            )
            matched_gt_indices = {gt_idx for _, gt_idx in tp_gt_pairs}
            gt_total += len(gts)
            tp_total += len(matched_gt_indices)
            fp_total += len(fp_indices)
            for idx, box in enumerate(gts):
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                if area < small_area_threshold:
                    gt_small += 1
                    if idx in matched_gt_indices:
                        tp_small += 1

    return {
        "track_recall_all": (tp_total / gt_total) if gt_total > 0 else float("nan"),
        "track_precision_all": (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 1.0,
        "track_recall_small": (tp_small / gt_small) if gt_small > 0 else float("nan"),
        "gt_small_count": int(gt_small),
        "tp_small_count": int(tp_small),
        "gt_total_count": int(gt_total),
        "tp_total_count": int(tp_total),
        "fp_total_count": int(fp_total),
    }


def _extract_trackeval_fields(summary: dict | None) -> dict[str, float]:
    m = summary or {}
    return {
        "HOTA": float(m.get("HOTA", np.nan)),
        "DetA": float(m.get("DetA", np.nan)),
        "AssA": float(m.get("AssA", np.nan)),
        "LocA": float(m.get("LocA", np.nan)),
        "MOTA": float(m.get("MOTA", np.nan)),
        "MOTP": float(m.get("MOTP", np.nan)),
        "IDF1": float(m.get("IDF1", np.nan)),
        "CLR_Re": float(m.get("CLR_Re", np.nan)),
        "CLR_Pr": float(m.get("CLR_Pr", np.nan)),
        "CLR_TP": float(m.get("CLR_TP", 0.0)),
        "CLR_FN": float(m.get("CLR_FN", 0.0)),
        "CLR_FP": float(m.get("CLR_FP", 0.0)),
        "IDSW": float(m.get("IDSW", 0.0)),
        "Frag": float(m.get("Frag", 0.0)),
        "MT": float(m.get("MT", 0.0)),
        "PT": float(m.get("PT", 0.0)),
        "ML": float(m.get("ML", 0.0)),
    }


def calibrate_trackers_and_roi_on_val(
    sequences: dict[str, list[dict[str, object]]],
    search_detector: DetectorLike,
    track_detector: DetectorLike,
    mot_dataset_dir: Path,
    ablation_dir: Path,
    *,
    trackers: tuple[str, ...] = ABLATION_TRACKERS,
    bytetrack_grid: dict[str, list[float | int]] | None = None,
    ocsort_grid: dict[str, list[float | int]] | None = None,
    lost_frames_candidates: tuple[int, ...] = DEFAULT_VAL_LOST_FRAMES,
    fps: int = 30,
    det_conf: float = 0.10,
    seed_conf: float = 0.25,
    roi_crop_size: int = 640,
    resume: bool = False,
    frame_loader: Callable[[dict[str, object]], np.ndarray] = _read_sequence_frame,
) -> dict[str, object]:
    """Calibrates ByteTrack/OC-SORT grids on cached 1280 detections and ROI hysteresis on val."""
    ablation_dir = Path(ablation_dir)
    mot_dataset_dir = Path(mot_dataset_dir)
    trackers_out_dir = ablation_dir / "trackers"
    reports_dir = ablation_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ablation_dir / "detections_cache" / "val"

    collect_openvino_detections_cache(
        sequences,
        search_detector,
        cache_dir,
        conf_thres=det_conf,
        resume=resume,
        frame_loader=frame_loader,
    )

    grids = {
        "bytetrack": _grid_combinations(bytetrack_grid or DEFAULT_BYTETRACK_GRID),
        "ocsort": _grid_combinations(ocsort_grid or DEFAULT_OCSORT_GRID),
    }

    grid_specs: list[tuple[str, str, dict[str, float | int]]] = []
    for tracker_name in trackers:
        for idx, cfg in enumerate(grids[tracker_name]):
            grid_specs.append((tracker_name, f"va_val_{tracker_name}_run_{idx}", cfg))

    pending_eval_runs: list[str] = []
    tracker_timings: dict[str, dict[str, float]] = {}
    for tracker_name, run_name, cfg in grid_specs:
        summary_file = trackers_out_dir / "drone-val" / run_name / "pedestrian_summary.txt"
        data_dir = trackers_out_dir / "drone-val" / run_name / "data"
        telemetry_file = trackers_out_dir / "drone-val" / run_name / "tracker_timing.json"
        if resume and summary_file.is_file() and telemetry_file.is_file():
            tracker_timings[run_name] = json.loads(telemetry_file.read_text(encoding="utf-8"))
            continue
        if not (resume and data_dir.is_dir() and any(data_dir.glob("*.txt"))):
            timing = run_tracking_pipeline(
                split_name="val",
                tracker_name=tracker_name,
                tracker_config=cfg,
                run_dir_name=run_name,
                dataset_base_dir=mot_dataset_dir,
                trackers_out_dir=trackers_out_dir,
                detections_source_dir=cache_dir,
            )
            telemetry_file.parent.mkdir(parents=True, exist_ok=True)
            telemetry_file.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        else:
            timing = (
                json.loads(telemetry_file.read_text(encoding="utf-8"))
                if telemetry_file.is_file()
                else {"avg_latency_ms": 0.0, "avg_fps": 0.0}
            )
        tracker_timings[run_name] = timing
        if not (resume and summary_file.is_file()):
            pending_eval_runs.append(run_name)

    if pending_eval_runs:
        run_trackeval("val", pending_eval_runs, mot_dataset_dir, trackers_out_dir)

    grid_rows: list[dict[str, object]] = []
    best_configs: dict[str, dict[str, float | int]] = {}
    for tracker_name, run_name, cfg in grid_specs:
        summary = parse_trackeval_summary(
            trackers_out_dir / "drone-val" / run_name / "pedestrian_summary.txt"
        )
        timing = tracker_timings.get(run_name, {})
        grid_rows.append({
            "run_name": run_name,
            "tracker": tracker_name,
            "config_json": json.dumps(cfg, sort_keys=True),
            **{f"param_{k}": v for k, v in cfg.items()},
            "tracker_avg_latency_ms": float(timing.get("avg_latency_ms", np.nan)),
            "tracker_avg_fps": float(timing.get("avg_fps", np.nan)),
            **_extract_trackeval_fields(summary),
        })

    val_grid_df = pd.DataFrame(grid_rows)
    for tracker_name in trackers:
        sub = val_grid_df[val_grid_df["tracker"] == tracker_name].dropna(subset=["HOTA"])
        if sub.empty:
            raise RuntimeError(f"No valid TrackEval val summary found for tracker={tracker_name!r}")
        best_row = sub.sort_values(["HOTA", "IDF1"], ascending=[False, False]).iloc[0]
        best_configs[tracker_name] = json.loads(str(best_row["config_json"]))

    roi_runs: list[tuple[str, int, str]] = []
    roi_telemetries: dict[str, dict[str, object]] = {}
    pending_roi_eval: list[str] = []
    for tracker_name in trackers:
        for max_lost in lost_frames_candidates:
            run_name = f"va_val_roi_kalman_{tracker_name}_lost{int(max_lost)}"
            roi_runs.append((tracker_name, int(max_lost), run_name))
            run_dir = trackers_out_dir / "drone-val" / run_name
            data_dir = run_dir / "data"
            telemetry_file = run_dir / "run_telemetry.json"
            summary_file = run_dir / "pedestrian_summary.txt"

            if resume and telemetry_file.is_file() and data_dir.is_dir() and any(data_dir.glob("*.txt")):
                telemetry = json.loads(telemetry_file.read_text(encoding="utf-8"))
            else:
                telemetry = run_adaptive_roi_on_sequences(
                    sequences,
                    search_detector=search_detector,
                    track_detector=track_detector,
                    config=AdaptiveROIConfig(
                        mode="roi_kalman",
                        tracker_name=tracker_name,
                        tracker_config=best_configs[tracker_name],
                        fps=int(fps),
                        roi_crop_size=int(roi_crop_size),
                        search_conf=float(det_conf),
                        track_conf=float(det_conf),
                        seed_conf=float(seed_conf),
                        max_lost_frames=int(max_lost),
                    ),
                    output_dir=data_dir,
                    frame_loader=frame_loader,
                )
                telemetry_file.parent.mkdir(parents=True, exist_ok=True)
                telemetry_file.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
            roi_telemetries[run_name] = telemetry
            if not (resume and summary_file.is_file()):
                pending_roi_eval.append(run_name)

    if pending_roi_eval:
        run_trackeval("val", pending_roi_eval, mot_dataset_dir, trackers_out_dir)

    seq_names = list(sequences.keys())
    roi_rows: list[dict[str, object]] = []
    for tracker_name, max_lost, run_name in roi_runs:
        run_dir = trackers_out_dir / "drone-val" / run_name
        summary = parse_trackeval_summary(run_dir / "pedestrian_summary.txt")
        small_metrics = compute_track_small_recall(
            mot_dataset_dir / "drone-val",
            run_dir / "data",
            seq_names,
        )
        tel = roi_telemetries[run_name]
        roi_rows.append({
            "run_name": run_name,
            "tracker": tracker_name,
            "max_lost_frames": max_lost,
            "e2e_fps": float(tel["e2e_fps"]),
            "e2e_mean_ms": float(tel["e2e_mean_ms"]),
            "e2e_p50_ms": float(tel["e2e_p50_ms"]),
            "e2e_p95_ms": float(tel["e2e_p95_ms"]),
            "pct_track_frames": float(tel["pct_track_frames"]),
            "search_transitions": int(tel["search_transitions"]),
            **_extract_trackeval_fields(summary),
            **small_metrics,
        })

    val_roi_df = pd.DataFrame(roi_rows)
    grouped_lost = (
        val_roi_df.groupby("max_lost_frames", as_index=False)[["HOTA", "e2e_fps"]]
        .mean()
        .sort_values(["HOTA", "e2e_fps"], ascending=[False, False])
    )
    best_max_lost_frames = int(grouped_lost.iloc[0]["max_lost_frames"])

    val_grid_csv = reports_dir / "val_tracker_grid.csv"
    val_roi_csv = reports_dir / "val_roi_calibration.csv"
    val_grid_df.to_csv(val_grid_csv, index=False)
    val_roi_df.to_csv(val_roi_csv, index=False)

    return {
        "best_tracker_configs": best_configs,
        "best_max_lost_frames": best_max_lost_frames,
        "val_grid_df": val_grid_df,
        "val_roi_df": val_roi_df,
        "val_grid_csv": val_grid_csv,
        "val_roi_csv": val_roi_csv,
    }


def run_video_ablation_on_test(
    sequences: dict[str, list[dict[str, object]]],
    search_detector: DetectorLike,
    track_detector: DetectorLike,
    mot_dataset_dir: Path,
    ablation_dir: Path,
    *,
    tracker_configs: dict[str, dict[str, float | int]],
    max_lost_frames: int = 4,
    modes: tuple[str, ...] = ABLATION_MODES,
    trackers: tuple[str, ...] = ABLATION_TRACKERS,
    eval_split: str = "test",
    fps: int = 30,
    det_conf: float = 0.10,
    seed_conf: float = 0.25,
    roi_crop_size: int = 640,
    resume: bool = False,
    metadata_extra: dict[str, object] | None = None,
    frame_loader: Callable[[dict[str, object]], np.ndarray] = _read_sequence_frame,
) -> dict[str, object]:
    """Runs Full 1280, Resize 640, ROI Static, and ROI Kalman across frozen trackers on test."""
    ablation_dir = Path(ablation_dir)
    mot_dataset_dir = Path(mot_dataset_dir)
    trackers_out_dir = ablation_dir / "trackers"
    reports_dir = ablation_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_folder = f"drone-{eval_split}"

    run_specs: list[tuple[str, str, str]] = []
    telemetries: dict[str, dict[str, object]] = {}
    pending_trackeval: list[str] = []

    for mode in modes:
        for tracker_name in trackers:
            if tracker_name not in tracker_configs:
                raise KeyError(f"Missing calibrated tracker config for {tracker_name!r}")
            run_name = f"va_{eval_split}_{mode}_{tracker_name}"
            run_specs.append((mode, tracker_name, run_name))
            run_dir = trackers_out_dir / split_folder / run_name
            data_dir = run_dir / "data"
            telemetry_file = run_dir / "run_telemetry.json"
            summary_file = run_dir / "pedestrian_summary.txt"

            if resume and telemetry_file.is_file() and data_dir.is_dir() and any(data_dir.glob("*.txt")):
                telemetry = json.loads(telemetry_file.read_text(encoding="utf-8"))
            else:
                telemetry = run_adaptive_roi_on_sequences(
                    sequences,
                    search_detector=search_detector,
                    track_detector=track_detector,
                    config=AdaptiveROIConfig(
                        mode=mode,
                        tracker_name=tracker_name,
                        tracker_config=tracker_configs[tracker_name],
                        fps=int(fps),
                        roi_crop_size=int(roi_crop_size),
                        search_conf=float(det_conf),
                        track_conf=float(det_conf),
                        seed_conf=float(seed_conf),
                        max_lost_frames=int(max_lost_frames),
                    ),
                    output_dir=data_dir,
                    frame_loader=frame_loader,
                )
                telemetry_file.parent.mkdir(parents=True, exist_ok=True)
                telemetry_file.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")

            telemetries[run_name] = telemetry
            if not (resume and summary_file.is_file()):
                pending_trackeval.append(run_name)

    if pending_trackeval:
        run_trackeval(eval_split, pending_trackeval, mot_dataset_dir, trackers_out_dir)

    seq_names = list(sequences.keys())
    rows: list[dict[str, object]] = []
    for mode, tracker_name, run_name in run_specs:
        run_dir = trackers_out_dir / split_folder / run_name
        summary = parse_trackeval_summary(run_dir / "pedestrian_summary.txt")
        small_metrics = compute_track_small_recall(
            mot_dataset_dir / split_folder,
            run_dir / "data",
            seq_names,
        )
        tel = telemetries[run_name]
        rows.append({
            "run_name": run_name,
            "mode": mode,
            "tracker": tracker_name,
            "e2e_fps": float(tel["e2e_fps"]),
            "e2e_mean_ms": float(tel["e2e_mean_ms"]),
            "e2e_p50_ms": float(tel["e2e_p50_ms"]),
            "e2e_p95_ms": float(tel["e2e_p95_ms"]),
            "total_frames": int(tel["total_frames"]),
            "search_frames": int(tel["search_frames"]),
            "track_frames": int(tel["track_frames"]),
            "pct_track_frames": float(tel["pct_track_frames"]),
            "search_to_track_transitions": int(tel["search_to_track_transitions"]),
            "track_to_search_transitions": int(tel["track_to_search_transitions"]),
            "search_transitions": int(tel["search_transitions"]),
            **_extract_trackeval_fields(summary),
            **small_metrics,
        })

    results_df = pd.DataFrame(rows)
    csv_path = reports_dir / f"{eval_split}_video_ablation.csv"
    metadata_path = reports_dir / "video_ablation_metadata.json"

    existing_meta: dict[str, object] = {}
    if metadata_path.is_file():
        existing_meta = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata = {
        **existing_meta,
        "eval_split": eval_split,
        "fps": int(fps),
        "det_conf": float(det_conf),
        "seed_conf": float(seed_conf),
        "roi_crop_size": int(roi_crop_size),
        "max_lost_frames": int(max_lost_frames),
        "tracker_configs": tracker_configs,
        "modes": list(modes),
        "trackers": list(trackers),
        "num_sequences": len(seq_names),
        "sequences": seq_names,
        **(metadata_extra or {}),
    }
    results_df.to_csv(csv_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "results_df": results_df,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def resolve_default_openvino_models(quantization_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolves default 1280 and 640 OpenVINO INT8 artifact directories."""
    qdir = Path(quantization_dir) if quantization_dir is not None else config.QUANTIZATION_DIR
    m1280 = get_quantized_model_paths(qdir, 1280)["openvino_int8"]
    m640 = get_quantized_model_paths(qdir, 640)["openvino_int8"]
    return m1280, m640


def build_openvino_detector_pair(
    model_1280_dir: Path,
    model_640_dir: Path,
    *,
    det_conf: float = 0.10,
) -> tuple[OpenVINOYOLODetector, OpenVINOYOLODetector]:
    """Compiles 1280 search and 640 track OpenVINO INT8 detectors once for ablation."""
    det_1280 = OpenVINOYOLODetector(model_1280_dir, imgsz=1280, conf_thres=det_conf)
    det_640 = OpenVINOYOLODetector(model_640_dir, imgsz=640, conf_thres=det_conf)
    return det_1280, det_640