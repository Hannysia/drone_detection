from collections.abc import Iterable
import logging
from pathlib import Path
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO
from src import config
from src.config import MANIFESTS_DIR, OUTPUT_DIR
from src.processing.dataset import parse_frame_filename_logic
from src.processing.manifests import select_manifest_frames
from src.processing.yolo_dataset import (
    _validate_run_name,
    cleanup_yolo_dataset,
    prepare_yolo_dataset,
)
from src.utils.geometry import calculate_iou_xyxy
from src.utils.visualization import plot_unified_calibration_curves

log = logging.getLogger(__name__)

def run_trackeval(
    split_name: str, 
    trackers_to_eval: list[str], 
    dataset_base_dir: Path, 
    trackers_out_dir: Path
) -> None:
    """Runs TrackEval silently, saving all results to disk without console printing."""
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        'PLOT_CURVES': False,
        'PRINT_RESULTS': False,       
        'PRINT_ONLY_COMBINED': True,
        'TIME_PROGRESS': False,       
        'PRINT_CONFIG': False         
    })
    
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        'GT_FOLDER': str(dataset_base_dir),
        'TRACKERS_FOLDER': str(trackers_out_dir),
        'SUB_SPLIT': '',
        'SPLIT_TO_EVAL': split_name,
        'BENCHMARK': 'drone',
        'TRACKERS_TO_EVAL': trackers_to_eval,
        'CLASSES_TO_EVAL': ['pedestrian'],
        'SKIP_SPLIT_FOL': False,
        'PRINT_CONFIG': False
    })
    
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity()
    ]
    
    print(f"Evaluating {len(trackers_to_eval)} configurations on split '{split_name}'...")
    evaluator.evaluate([dataset], metrics)
    print("Evaluation complete. Results saved to disk successfully.")

def parse_trackeval_summary(summary_path: Path) -> dict | None:
    """Parses TrackEval summary file and returns a dictionary of metrics."""
    if not summary_path.exists():
        return None
        
    with open(summary_path, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if len(lines) < 2:
        return None
        
    headers = lines[0].split()
    values = lines[1].split()
    normalized_headers = [h.replace('(%)', '') for h in headers]
    
    metrics = {}
    for h, v in zip(normalized_headers, values):
        try:
            metrics[h] = float(v)
        except ValueError:
            metrics[h] = v
    return metrics



def find_and_display_best_configs(
    trackers_out_dir: Path,
    bytetrack_combinations: list[dict],
    ocsort_combinations: list[dict],
    split_name: str = "val",
    suffix: str = ""
) -> tuple[dict, dict]:
    """Parses TrackEval summaries for all combinations and displays the best configured tracker metrics."""
    best_bt_hota, best_bt_metrics, best_bt_config = -1.0, None, None
    for idx, config in enumerate(bytetrack_combinations):
        summary_file = trackers_out_dir / f"drone-{split_name}" / f"bytetrack{suffix}_run_{idx}" / "pedestrian_summary.txt"
        metrics = parse_trackeval_summary(summary_file)
        if metrics and metrics.get("HOTA", 0.0) > best_bt_hota:
            best_bt_hota = metrics["HOTA"]
            best_bt_metrics, best_bt_config = metrics, config
                
    best_oc_hota, best_oc_metrics, best_oc_config = -1.0, None, None
    for idx, config in enumerate(ocsort_combinations):
        summary_file = trackers_out_dir / f"drone-{split_name}" / f"ocsort{suffix}_run_{idx}" / "pedestrian_summary.txt"
        metrics = parse_trackeval_summary(summary_file)
        if metrics and metrics.get("HOTA", 0.0) > best_oc_hota:
            best_oc_hota = metrics["HOTA"]
            best_oc_metrics, best_oc_config = metrics, config

    if best_bt_metrics or best_oc_metrics:
        print("\n" + "="*115)
        print(f"                 BEST CALIBRATED CONFIGURATIONS ON {split_name.upper()} SPLIT")
        print("="*115)
        comparison_data = []
        for name, m, cfg in [("ByteTrack", best_bt_metrics, best_bt_config), ("OC-SORT", best_oc_metrics, best_oc_config)]:
            if m:
                comparison_data.append({
                    "Tracker": name, 
                    "HOTA": m.get("HOTA", 0.0), 
                    "IDF1": m.get("IDF1", 0.0), 
                    "MOTA": m.get("MOTA", 0.0),
                    "FP": int(m.get("CLR_FP", 0)), 
                    "FN": int(m.get("CLR_FN", 0)), 
                    "IDSW": int(m.get("IDSW", 0))
                })
        df = pd.DataFrame(comparison_data)
        float_format = {"HOTA": "{:.2f}%", "IDF1": "{:.2f}%", "MOTA": "{:.2f}%"}
        
        try:
            from IPython.display import display
            display(df.style.format(float_format).hide(axis='index'))
        except ImportError:
            print(df.to_string(index=False))
            
        print("="*115 + "\n")
        
    return best_bt_config, best_oc_config



def evaluate_model(
    weights_path: Path,
    yaml_path: Path,
    run_name: str,
    *,
    split: str = "test",
    imgsz: int = 1280,
    batch: int = 16,
    conf: float | None = None,
    plots: bool = True,
) -> dict[str, float]:
    """Evaluates the trained model on the target split and returns metrics."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    log.info("\n" + "=" * 80)
    log.info("STARTING YOLO EVALUATION ON %s DATA: %s", split.upper(), run_name.upper())
    log.info("=" * 80)
    log.info("Loading weights from: %s", weights_path.resolve())

    model = YOLO(str(weights_path))
    val_kwargs = {
        "data": str(yaml_path),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "rect": False,
        "project": str(OUTPUT_DIR),
        "name": f"{run_name}_{split}_evaluation",
        "exist_ok": True,
        "save": True,
        "plots": plots,
    }
    if conf is not None:
        val_kwargs["conf"] = conf

    metrics = model.val(**val_kwargs)
    metrics_dict = (
        dict(metrics.results_dict)
        if metrics is not None and hasattr(metrics, "results_dict")
        else {}
    )
    if metrics_dict:
        log.info("\n" + "=" * 50)
        log.info("%s DATA METRICS REPORT (%s)", split.upper(), run_name.upper())
        log.info("=" * 50)
        log.info("Precision (P):              %.4f", metrics_dict.get("metrics/precision(B)", 0.0))
        log.info("Recall (R):                 %.4f", metrics_dict.get("metrics/recall(B)", 0.0))
        log.info("mAP50:                      %.4f", metrics_dict.get("metrics/mAP50(B)", 0.0))
        log.info("mAP50-95:                   %.4f", metrics_dict.get("metrics/mAP50-95(B)", 0.0))
        log.info("=" * 50)

    return metrics_dict


def load_eval_model(weights_path: str, device: torch.device) -> tuple[YOLO, torch.nn.Module]:
    """Loads a trained YOLO model and prepares its PyTorch module for inference."""
    model = YOLO(weights_path)
    pytorch_model = model.model
    pytorch_model.eval()
    pytorch_model.to(device)
    log.info(f"[EVAL] Model loaded from: {weights_path}")
    log.info(f"[EVAL] Parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    return model, pytorch_model


def get_detections_boxes(
    model: YOLO,
    img_path: str,
    conf: float,
    imgsz: int = 1280,
) -> list[tuple[int, int, int, int, float]]:
    """Executes standard YOLO inference to retrieve bounding boxes."""
    results = model(img_path, conf=conf, imgsz=imgsz, rect=False, verbose=False)
    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf_val))
    return boxes


def _greedy_match_predictions(
    gt_boxes: list,
    pred_boxes: list,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int], dict[int, float]]:
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes))), {}
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), [], {}

    ordered_preds = sorted(
        enumerate(pred_boxes),
        key=lambda item: item[1][4],
        reverse=True,
    )

    matched_gts: set[int] = set()
    matched_gt_ious: dict[int, float] = {}
    tp_gt_pairs: list[tuple[int, int]] = []
    fp_indices: list[int] = []

    for orig_idx, pred in ordered_preds:
        best_iou = -1.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gts:
                continue
            iou = calculate_iou_xyxy(pred[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            matched_gts.add(best_gt_idx)
            matched_gt_ious[best_gt_idx] = float(best_iou)
            tp_gt_pairs.append((orig_idx, best_gt_idx))
        else:
            fp_indices.append(orig_idx)

    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gts]
    return tp_gt_pairs, fp_indices, fn_indices, matched_gt_ious


def _match_predictions_with_details(
    gt_boxes: list,
    pred_boxes: list,
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int], list[int], dict[int, float], set[int]]:
    tp_gt_pairs, fp_indices, fn_indices, matched_gt_ious = _greedy_match_predictions(
        gt_boxes,
        pred_boxes,
        iou_threshold=iou_threshold,
    )
    tp_indices = [pred_idx for pred_idx, _ in tp_gt_pairs]
    return tp_indices, fp_indices, fn_indices, matched_gt_ious, set(matched_gt_ious.keys())


def evaluate_predictions(
    gt_boxes: list,
    pred_boxes: list,
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:
    """Greedy IoU matching returning indices relative to the input pred_boxes list."""
    tp_indices, fp_indices, fn_indices, _, _ = _match_predictions_with_details(
        gt_boxes,
        pred_boxes,
        iou_threshold=iou_threshold,
    )
    return tp_indices, fp_indices, fn_indices


def _compute_f_scores(precision: float, recall: float) -> tuple[float, float]:
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
    return f1, f05


def _compute_prf_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1, f05 = _compute_f_scores(precision, recall)
    return precision, recall, f1, f05


def run_dataset_predictions_db(
    model: YOLO,
    dataset_root: Path,
    imgsz: int = 1280,
    *,
    collection_conf: float,
    split: str = "val",
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    allowed_splits: tuple[str, ...] = ("val", "test"),
) -> list[dict]:
    """Collects predictions for manifest-selected frames using collection_conf."""
    selected_frames = select_manifest_frames(
        Path(dataset_root),
        split=split,
        manifest_path=manifest_path,
        source_path=source_path,
        allowed_splits=allowed_splits,
    )

    log.info("Found %d active manifest images for split %r.", len(selected_frames), split)

    device = 0 if torch.cuda.is_available() else "cpu"
    raw_database = []

    for img_path, category in tqdm(selected_frames, desc="Running Evaluation Inference"):
        is_positive = category == "positive"
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Unreadable evaluation image: {img_path}")

        h_orig, w_orig = img.shape[:2]
        gt_boxes = []

        if is_positive:
            txt_path = img_path.with_suffix(".txt")

            with txt_path.open(encoding="utf-8") as labels:
                for line_no, line in enumerate(labels, start=1):
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        raise ValueError(f"{txt_path}:{line_no}: expected 5 YOLO fields")

                    try:
                        x_c, y_c, w, h = map(float, parts[1:])
                    except ValueError as exc:
                        raise ValueError(
                            f"{txt_path}:{line_no}: invalid box coordinates"
                        ) from exc

                    gt_boxes.append([
                        (x_c - w / 2) * w_orig,
                        (y_c - h / 2) * h_orig,
                        (x_c + w / 2) * w_orig,
                        (y_c + h / 2) * h_orig,
                    ])

            if not gt_boxes:
                raise ValueError(f"Positive image has no annotated boxes: {txt_path}")

        results = model.predict(
            img,
            imgsz=imgsz,
            conf=collection_conf,
            verbose=False,
            device=device,
            rect=False,
        )[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                pred_boxes.append([coords[0], coords[1], coords[2], coords[3], conf])

        raw_database.append({
            "filename": img_path.name,
            "path": img_path,
            "is_positive": is_positive,
            "width": w_orig,
            "height": h_orig,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
        })

    return raw_database


def download_yolo_run_artifacts(
    run_names: Iterable[str],
    *,
    cache_dir: Path | None = None,
    model_prefix: str = "yolo26n",
    repo_id: str | None = None,
    token: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Downloads best weights and results.csv for YOLO training runs from Hugging Face."""
    target_repo = config.REPO_ID if repo_id is None else repo_id
    target_token = config.HF_TOKEN if token is None else token
    if not isinstance(target_repo, str) or not target_repo.strip():
        raise ValueError("REPO_ID is required for Hugging Face operations")
    if not isinstance(target_token, str) or not target_token.strip():
        raise ValueError("HF_TOKEN is required for Hugging Face operations")

    validated_prefix = _validate_run_name(model_prefix)
    local_dir = Path(cache_dir) if cache_dir is not None else config.WORKING_DIR / "hf_artifacts"
    local_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    for run_name in run_names:
        validated_name = _validate_run_name(run_name)
        weights = Path(hf_hub_download(
            repo_id=target_repo.strip(),
            filename=f"models/{validated_prefix}_{validated_name}_best.pt",
            repo_type="dataset",
            token=target_token.strip(),
            local_dir=local_dir,
        ))
        results_csv = Path(hf_hub_download(
            repo_id=target_repo.strip(),
            filename=f"runs/{validated_name}/results.csv",
            repo_type="dataset",
            token=target_token.strip(),
            local_dir=local_dir,
        ))
        artifacts[validated_name] = {"weights": weights, "results_csv": results_csv}

    if not artifacts:
        raise ValueError("Expected at least one run name to download")
    return artifacts


def calibrate_confidence_thresholds(
    raw_database: list[dict],
    *,
    thresholds: np.ndarray | None = None,
    iou_threshold: float = 0.5,
    plot: bool = True,
    save_path: Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Sweeps confidence thresholds on a validation DB to maximize F0.5 and F1."""
    if not raw_database:
        raise ValueError("Calibration database is empty")

    grid = np.linspace(0.05, 0.95, 37) if thresholds is None else np.asarray(thresholds, dtype=float)
    if grid.size == 0:
        raise ValueError("Threshold grid must not be empty")

    precision_curve, recall_curve, f1_curve, f05_curve, neg_fa_rate = [], [], [], [], []
    for threshold in grid:
        tps_count, fps_count, fns_count = 0, 0, 0
        total_negatives, negatives_with_false_alarms = 0, 0

        for entry in raw_database:
            filtered_preds = [pred for pred in entry["pred_boxes"] if pred[4] >= threshold]
            tp, fp, fn = evaluate_predictions(
                entry["gt_boxes"],
                filtered_preds,
                iou_threshold=iou_threshold,
            )
            tps_count += len(tp)
            fps_count += len(fp)
            fns_count += len(fn)

            if not entry["is_positive"]:
                total_negatives += 1
                if filtered_preds:
                    negatives_with_false_alarms += 1

        precision, recall, f1, f05 = _compute_prf_metrics(tps_count, fps_count, fns_count)
        fa_rate = (negatives_with_false_alarms / total_negatives * 100) if total_negatives > 0 else 0.0

        precision_curve.append(precision)
        recall_curve.append(recall)
        f1_curve.append(f1)
        f05_curve.append(f05)
        neg_fa_rate.append(fa_rate)

    opt_f05_idx = int(np.argmax(f05_curve))
    opt_f1_idx = int(np.argmax(f1_curve))
    opt_threshold_f05 = float(grid[opt_f05_idx])
    opt_threshold_f1 = float(grid[opt_f1_idx])

    summary_df = pd.DataFrame([
        {
            "Target": "Max F0.5 (Homing)",
            "Threshold": round(opt_threshold_f05, 4),
            "Precision": round(precision_curve[opt_f05_idx], 4),
            "Recall": round(recall_curve[opt_f05_idx], 4),
            "Score": round(f05_curve[opt_f05_idx], 4),
            "Neg FA Rate (%)": round(neg_fa_rate[opt_f05_idx], 2),
        },
        {
            "Target": "Max F1 (Balanced)",
            "Threshold": round(opt_threshold_f1, 4),
            "Precision": round(precision_curve[opt_f1_idx], 4),
            "Recall": round(recall_curve[opt_f1_idx], 4),
            "Score": round(f1_curve[opt_f1_idx], 4),
            "Neg FA Rate (%)": round(neg_fa_rate[opt_f1_idx], 2),
        },
    ])

    if plot:
        plot_unified_calibration_curves(
            grid,
            precision_curve,
            recall_curve,
            f1_curve,
            f05_curve,
            neg_fa_rate,
            opt_threshold_f1,
            opt_threshold_f05,
            save_path=save_path,
            show=show,
        )

    return {
        "opt_threshold_f05": opt_threshold_f05,
        "opt_threshold_f1": opt_threshold_f1,
        "summary": summary_df,
        "thresholds": grid,
        "precision": precision_curve,
        "recall": recall_curve,
        "f1": f1_curve,
        "f05": f05_curve,
        "neg_fa_rate": neg_fa_rate,
    }


def evaluate_runs_on_test(
    runs: dict[str, Path],
    dataset_root: Path,
    *,
    calibrated_run: str | None = None,
    calibrated_conf: float | None = None,
    imgsz: int = 1280,
    batch: int = 16,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    plots: bool = True,
) -> pd.DataFrame:
    """Stages manifest-curated data and evaluates multiple runs on the locked test split."""
    if not runs:
        raise ValueError("Expected at least one run to evaluate")
    if (calibrated_run is None) != (calibrated_conf is None):
        raise ValueError("Provide both calibrated_run and calibrated_conf together")
    if calibrated_run is not None and calibrated_run not in runs:
        raise ValueError(f"Unknown calibrated_run: {calibrated_run!r}")

    working_dir = config.WORKING_DIR
    eval_yaml = prepare_yolo_dataset(
        Path(dataset_root),
        working_dir,
        "eval_v1",
        use_negatives=True,
        manifest_path=manifest_path,
        source_path=source_path,
        overwrite=True,
    )

    rows = []
    try:
        for run_name, weights_path in runs.items():
            validated_name = _validate_run_name(run_name)
            metrics = evaluate_model(
                Path(weights_path),
                eval_yaml,
                validated_name,
                split="test",
                imgsz=imgsz,
                batch=batch,
                plots=plots,
            )
            precision = float(metrics.get("metrics/precision(B)", 0.0))
            recall = float(metrics.get("metrics/recall(B)", 0.0))
            _, f05 = _compute_f_scores(precision, recall)
            rows.append({
                "Run": validated_name,
                "Conf": "default (mAP)",
                "Precision": precision,
                "Recall": recall,
                "F0.5": f05,
                "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            })

        if calibrated_run is not None and calibrated_conf is not None:
            validated_cal = _validate_run_name(calibrated_run)
            cal_metrics = evaluate_model(
                Path(runs[calibrated_run]),
                eval_yaml,
                f"{validated_cal}_cal_f05",
                split="test",
                imgsz=imgsz,
                batch=batch,
                conf=float(calibrated_conf),
                plots=plots,
            )
            cal_precision = float(cal_metrics.get("metrics/precision(B)", 0.0))
            cal_recall = float(cal_metrics.get("metrics/recall(B)", 0.0))
            _, cal_f05 = _compute_f_scores(cal_precision, cal_recall)
            rows.append({
                "Run": validated_cal,
                "Conf": f"F0.5 ({float(calibrated_conf):.4f})",
                "Precision": cal_precision,
                "Recall": cal_recall,
                "F0.5": cal_f05,
                "mAP50": np.nan,
                "mAP50-95": np.nan,
            })
    finally:
        cleanup_yolo_dataset(working_dir, "eval_v1")

    return pd.DataFrame(rows)


def _classify_bbox_scale(box: list[float]) -> str:
    area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _classify_spatial_region(
    box: list[float],
    width: int,
    height: int,
    center_radius: float,
) -> str:
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive frame dimensions, got {width}x{height}")
    cx_norm = ((float(box[0]) + float(box[2])) / 2.0) / width
    cy_norm = ((float(box[1]) + float(box[3])) / 2.0) / height
    if abs(cx_norm - 0.5) <= center_radius and abs(cy_norm - 0.5) <= center_radius:
        return "Center"
    return "Off-center"


def _aggregate_gt_bucket_table(
    records: list[dict],
    group_key: str,
    col_label: str,
    ordered_groups: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(records)
    groups = ordered_groups if ordered_groups is not None else tuple(sorted(df[group_key].unique()))
    rows = []
    for name in groups:
        sub = df[df[group_key] == name] if not df.empty else df
        gt_count = int(len(sub))
        tp_count = int(sub["matched"].sum()) if gt_count else 0
        fn_count = gt_count - tp_count
        ious = sub.loc[sub["matched"], "iou"].dropna().tolist() if tp_count else []
        rows.append({
            col_label: name,
            "GT": gt_count,
            "TP": tp_count,
            "FN": fn_count,
            "Recall": tp_count / gt_count if gt_count else np.nan,
            "Median TP IoU": float(np.median(ious)) if ious else np.nan,
        })
    return pd.DataFrame(rows)


def _aggregate_metadata_table(frame_records: list[dict], key: str) -> pd.DataFrame:
    df = pd.DataFrame(frame_records)
    rows = []
    for group_value in sorted(df[key].unique()):
        sub = df[df[key] == group_value]
        pos_sub = sub[sub["is_positive"]]
        neg_sub = sub[~sub["is_positive"]]
        pos_gt = int(pos_sub["gt_count"].sum())
        tp_count = int(pos_sub["tp_count"].sum())
        fn_count = int(pos_sub["fn_count"].sum())
        pos_fp = int(pos_sub["fp_count"].sum())
        neg_frames = int(len(neg_sub))
        neg_fa_frames = int((neg_sub["fp_count"] > 0).sum())
        rows.append({
            "Group": group_value,
            "Frames": int(len(sub)),
            "Pos GT": pos_gt,
            "TP": tp_count,
            "FN": fn_count,
            "Recall": tp_count / pos_gt if pos_gt else np.nan,
            "Pos FP": pos_fp,
            "Neg Frames": neg_frames,
            "Neg FA Frames": neg_fa_frames,
            "Neg FA Rate (%)": round(neg_fa_frames / neg_frames * 100, 2) if neg_frames else np.nan,
        })
    return pd.DataFrame(rows)


def analyze_model_errors(
    raw_database: list[dict],
    *,
    conf_threshold: float,
    iou_threshold: float = 0.5,
    loc_iou_threshold: float = 0.1,
    center_radius: float = 0.15,
    top_k: int = 6,
) -> dict[str, object]:
    """Computes stratified error analysis across target scale, metadata, position, and error modes."""
    if not raw_database:
        raise ValueError("Evaluation database is empty")
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError(f"Invalid conf_threshold: {conf_threshold}")
    if not 0.0 <= loc_iou_threshold <= iou_threshold <= 1.0:
        raise ValueError("Expected 0 <= loc_iou_threshold <= iou_threshold <= 1")
    if not 0.0 < center_radius <= 0.5:
        raise ValueError("center_radius must be in (0, 0.5]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    error_counts = {
        "TP": 0,
        "Background FP": 0,
        "Localization FP": 0,
        "Duplicate FP": 0,
        "Threshold Miss FN": 0,
        "Blind Miss FN": 0,
    }
    gt_records: list[dict] = []
    frame_records: list[dict] = []
    fp_cases: list[dict] = []
    fn_cases: list[dict] = []

    for entry in raw_database:
        gt_boxes = entry["gt_boxes"]
        preds_above = [pred for pred in entry["pred_boxes"] if pred[4] >= conf_threshold]
        preds_below = [pred for pred in entry["pred_boxes"] if pred[4] < conf_threshold]
        tp_idx, fp_idx, fn_idx, matched_gt_ious, matched_gts = _match_predictions_with_details(
            gt_boxes,
            preds_above,
            iou_threshold=iou_threshold,
        )
        error_counts["TP"] += len(tp_idx)

        for idx in fp_idx:
            pred_box = preds_above[idx][:4]
            best_any = max((calculate_iou_xyxy(pred_box, gt) for gt in gt_boxes), default=0.0)
            best_matched = max(
                (calculate_iou_xyxy(pred_box, gt_boxes[g]) for g in matched_gts),
                default=0.0,
            )
            if not gt_boxes or best_any < loc_iou_threshold:
                error_counts["Background FP"] += 1
            elif best_matched >= iou_threshold:
                error_counts["Duplicate FP"] += 1
            else:
                error_counts["Localization FP"] += 1

        width = int(entry.get("width", 0))
        height = int(entry.get("height", 0))
        for gt_idx, gt_box in enumerate(gt_boxes):
            is_matched = gt_idx in matched_gts
            if not is_matched:
                has_subthresh = any(
                    calculate_iou_xyxy(pred[:4], gt_box) >= iou_threshold
                    for pred in preds_below
                )
                if has_subthresh:
                    error_counts["Threshold Miss FN"] += 1
                else:
                    error_counts["Blind Miss FN"] += 1

            gt_records.append({
                "scale": _classify_bbox_scale(gt_box),
                "region": _classify_spatial_region(gt_box, width, height, center_radius),
                "matched": is_matched,
                "iou": matched_gt_ious.get(gt_idx, np.nan),
            })

        meta = parse_frame_filename_logic(entry["filename"])
        frame_records.append({
            "is_positive": bool(entry["is_positive"]),
            "background": meta["background"],
            "camera": meta["camera"],
            "shape": meta["shape"] if entry["is_positive"] else "no_drone",
            "gt_count": len(gt_boxes),
            "tp_count": len(tp_idx),
            "fn_count": len(fn_idx),
            "fp_count": len(fp_idx),
        })

        entry_path = Path(entry.get("path", entry["filename"]))
        if fp_idx:
            fp_boxes = [preds_above[i] for i in fp_idx]
            fp_cases.append({
                "path": entry_path,
                "filename": entry["filename"],
                "is_positive": bool(entry["is_positive"]),
                "n_fps": len(fp_boxes),
                "max_conf": max(float(b[4]) for b in fp_boxes),
                "fp_boxes": fp_boxes,
                "gt_boxes": gt_boxes,
            })

        if fn_idx:
            missed_boxes = [gt_boxes[i] for i in fn_idx]
            max_area = max(
                max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
                for b in missed_boxes
            )
            subthresh_hits = [
                pred
                for pred in preds_below
                if any(calculate_iou_xyxy(pred[:4], gt) >= iou_threshold for gt in missed_boxes)
            ]
            fn_cases.append({
                "path": entry_path,
                "filename": entry["filename"],
                "n_fns": len(missed_boxes),
                "max_missed_area": max_area,
                "fn_boxes": missed_boxes,
                "subthresh_boxes": subthresh_hits,
            })

    fp_cases.sort(key=lambda item: (item["max_conf"], item["n_fps"]), reverse=True)
    fn_cases.sort(key=lambda item: (item["n_fns"], item["max_missed_area"]), reverse=True)

    error_types_df = pd.DataFrame([
        {"Type": name, "Count": count}
        for name, count in error_counts.items()
    ])
    by_size_df = _aggregate_gt_bucket_table(
        gt_records,
        "scale",
        "Scale",
        ordered_groups=("small", "medium", "large"),
    )
    by_shape_df = _aggregate_metadata_table(frame_records, "shape")

    tp_total = error_counts["TP"]
    fp_total = (
        error_counts["Background FP"]
        + error_counts["Localization FP"]
        + error_counts["Duplicate FP"]
    )
    fn_total = error_counts["Threshold Miss FN"] + error_counts["Blind Miss FN"]
    precision, recall, f1, f05 = _compute_prf_metrics(tp_total, fp_total, fn_total)
    size_recalls = dict(zip(by_size_df["Scale"], by_size_df["Recall"]))
    neg_rows = by_shape_df[by_shape_df["Group"] == "no_drone"]
    neg_fpr = float(neg_rows["Neg FA Rate (%)"].iloc[0]) if not neg_rows.empty else float("nan")
    summary_df = pd.DataFrame([{
        "Conf": round(float(conf_threshold), 4),
        "Precision": precision,
        "Recall": recall,
        "F0.5": f05,
        "F1": f1,
        "Recall_small": float(size_recalls.get("small", np.nan)),
        "Recall_medium": float(size_recalls.get("medium", np.nan)),
        "Recall_large": float(size_recalls.get("large", np.nan)),
        "Neg FA Rate (%)": neg_fpr,
    }])

    return {
        "conf_threshold": float(conf_threshold),
        "summary": summary_df,
        "error_types": error_types_df,
        "by_size": by_size_df,
        "by_position": _aggregate_gt_bucket_table(
            gt_records,
            "region",
            "Region",
            ordered_groups=("Center", "Off-center"),
        ),
        "by_background": _aggregate_metadata_table(frame_records, "background"),
        "by_camera": _aggregate_metadata_table(frame_records, "camera"),
        "by_shape": by_shape_df,
        "worst_fps": fp_cases[:top_k],
        "worst_fns": fn_cases[:top_k],
    }


def _compute_average_precision(scores: np.ndarray, tp_flags: np.ndarray, total_gt: int) -> float:
    if total_gt <= 0:
        return float("nan")
    if scores.size == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_tp = tp_flags[order].astype(float)

    tp_cum = np.cumsum(sorted_tp)
    fp_cum = np.cumsum(1.0 - sorted_tp)
    keep = np.r_[np.diff(sorted_scores) < 0.0, True]

    recalls = tp_cum[keep] / float(total_gt)
    precisions = tp_cum[keep] / np.maximum(tp_cum[keep] + fp_cum[keep], 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _collect_matched_detections(
    raw_database: list[dict],
    iou_threshold: float,
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_scale_counts = {"small": 0, "medium": 0, "large": 0}
    det_scores: list[float] = []
    det_tps: list[int] = []
    det_scales: list[str] = []
    neg_max_confs: list[float] = []

    for entry in raw_database:
        gt_boxes = entry["gt_boxes"]
        pred_boxes = entry["pred_boxes"]
        for gt_box in gt_boxes:
            gt_scale_counts[_classify_bbox_scale(gt_box)] += 1

        tp_gt_pairs, fp_idx, _, _ = _greedy_match_predictions(
            gt_boxes,
            pred_boxes,
            iou_threshold=iou_threshold,
        )
        for pred_idx, gt_idx in tp_gt_pairs:
            det_scores.append(float(pred_boxes[pred_idx][4]))
            det_tps.append(1)
            det_scales.append(_classify_bbox_scale(gt_boxes[gt_idx]))
        for pred_idx in fp_idx:
            pred = pred_boxes[pred_idx]
            det_scores.append(float(pred[4]))
            det_tps.append(0)
            det_scales.append(_classify_bbox_scale(pred[:4]))

        if not entry["is_positive"]:
            neg_max_confs.append(max((float(pred[4]) for pred in pred_boxes), default=-1.0))

    return (
        gt_scale_counts,
        np.asarray(det_scores, dtype=float),
        np.asarray(det_tps, dtype=int),
        np.asarray(det_scales, dtype=object),
        np.asarray(neg_max_confs, dtype=float),
    )


def evaluate_threshold_free_slices(
    raw_database: list[dict],
    *,
    iou_threshold: float = 0.5,
    target_recall: float = 0.85,
    max_neg_fpr: float = 5.0,
    thresholds: np.ndarray | None = None,
) -> dict[str, object]:
    """Computes threshold-free AP slices and negative-frame false alarm trade-offs."""
    if not raw_database:
        raise ValueError("Evaluation database is empty")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"Invalid iou_threshold: {iou_threshold}")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"Invalid target_recall: {target_recall}")
    if not 0.0 <= max_neg_fpr <= 100.0:
        raise ValueError(f"Invalid max_neg_fpr: {max_neg_fpr}")

    grid = np.linspace(0.01, 0.99, 99) if thresholds is None else np.asarray(thresholds, dtype=float)
    if grid.size == 0:
        raise ValueError("Threshold grid must not be empty")

    gt_scale_counts, scores_arr, tps_arr, scales_arr, neg_arr = _collect_matched_detections(
        raw_database,
        iou_threshold,
    )
    scale_names = ("small", "medium", "large")
    total_gt = sum(gt_scale_counts.values())
    small_gt = gt_scale_counts["small"]
    total_negatives = int(neg_arr.size)

    ap50_all = _compute_average_precision(scores_arr, tps_arr, total_gt)
    ap50_by_scale = {}
    for name in scale_names:
        scale_mask = scales_arr == name
        ap50_by_scale[name] = _compute_average_precision(
            scores_arr[scale_mask],
            tps_arr[scale_mask],
            gt_scale_counts[name],
        )

    precision_curve, recall_curve, recall_small_curve = [], [], []
    f1_curve, f05_curve, neg_fa_rate = [], [], []
    small_tp_mask = (tps_arr == 1) & (scales_arr == "small")

    for threshold in grid:
        mask = scores_arr >= threshold
        tp_t = int(tps_arr[mask].sum())
        fp_t = int(mask.sum()) - tp_t
        fn_t = total_gt - tp_t
        precision, recall, f1, f05 = _compute_prf_metrics(tp_t, fp_t, fn_t)

        r_small = float((small_tp_mask & mask).sum() / small_gt) if small_gt > 0 else float("nan")
        neg_fpr = float(np.mean(neg_arr >= threshold) * 100.0) if total_negatives > 0 else float("nan")

        precision_curve.append(precision)
        recall_curve.append(recall)
        recall_small_curve.append(r_small)
        f1_curve.append(f1)
        f05_curve.append(f05)
        neg_fa_rate.append(neg_fpr)

    opt_f1_idx = int(np.argmax(f1_curve))
    opt_f05_idx = int(np.argmax(f05_curve))

    recall_ok_indices = [i for i, r in enumerate(recall_curve) if r >= target_recall]
    neg_fpr_at_recall = (
        float(min(neg_fa_rate[i] for i in recall_ok_indices))
        if total_negatives > 0 and recall_ok_indices
        else float("nan")
    )

    fpr_ok_indices = (
        [i for i, fpr in enumerate(neg_fa_rate) if fpr <= max_neg_fpr]
        if total_negatives > 0
        else []
    )
    recall_small_at_fpr = (
        float(max(recall_small_curve[i] for i in fpr_ok_indices))
        if small_gt > 0 and fpr_ok_indices
        else float("nan")
    )

    by_scale_rows = []
    for name in scale_names:
        gt_count = gt_scale_counts[name]
        tp_scale_mask = (tps_arr == 1) & (scales_arr == name)
        max_recall = float(tp_scale_mask.sum() / gt_count) if gt_count > 0 else float("nan")
        r_f1 = float((tp_scale_mask & (scores_arr >= grid[opt_f1_idx])).sum() / gt_count) if gt_count > 0 else float("nan")
        r_at_fpr = (
            float(max((tp_scale_mask & (scores_arr >= grid[i])).sum() / gt_count for i in fpr_ok_indices))
            if gt_count > 0 and fpr_ok_indices
            else float("nan")
        )
        by_scale_rows.append({
            "Scale": name,
            "GT": gt_count,
            "AP50": ap50_by_scale[name],
            "Max Recall (raw)": max_recall,
            "Recall @ Max-F1": r_f1,
            f"Recall @ Neg FPR<={max_neg_fpr:g}%": r_at_fpr,
        })

    mean_neg_fpr = float(np.mean(neg_fa_rate)) if total_negatives > 0 else float("nan")
    summary_df = pd.DataFrame([{
        "AP50": ap50_all,
        "AP50_small": ap50_by_scale["small"],
        "AP50_medium": ap50_by_scale["medium"],
        "AP50_large": ap50_by_scale["large"],
        "Max-F1 P": precision_curve[opt_f1_idx],
        "Max-F1 R": recall_curve[opt_f1_idx],
        "Max-F1 Recall_small": recall_small_curve[opt_f1_idx],
        "Max-F1 Neg FA Rate (%)": neg_fa_rate[opt_f1_idx],
        f"Recall_small @ Neg FPR<={max_neg_fpr:g}%": recall_small_at_fpr,
        f"Neg FPR @ R>={target_recall:g} (%)": neg_fpr_at_recall,
        "Mean Neg FA Rate (%)": mean_neg_fpr,
    }])

    return {
        "summary": summary_df,
        "by_scale": pd.DataFrame(by_scale_rows),
        "ap50": ap50_all,
        "ap50_by_scale": ap50_by_scale,
        "thresholds": grid,
        "precision": precision_curve,
        "recall": recall_curve,
        "recall_small": recall_small_curve,
        "f1": f1_curve,
        "f05": f05_curve,
        "neg_fa_rate": neg_fa_rate,
        "oracle_f1_conf": float(grid[opt_f1_idx]),
        "oracle_f05_conf": float(grid[opt_f05_idx]),
    }