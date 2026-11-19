import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.config import MANIFESTS_DIR, OUTPUT_DIR
from src.processing.manifests import select_manifest_frames
from src.utils.geometry import calculate_iou_xyxy

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
    results = model(img_path, conf=conf, imgsz=imgsz, verbose=False)
    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf_val))
    return boxes


def evaluate_predictions(
    gt_boxes: list,
    pred_boxes: list,
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:
    """Greedy IoU matching returning indices relative to the input pred_boxes list."""
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    ordered_preds = sorted(
        enumerate(pred_boxes),
        key=lambda item: item[1][4],
        reverse=True,
    )

    matched_gts = set()
    tp_indices = []
    fp_indices = []

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
            tp_indices.append(orig_idx)
        else:
            fp_indices.append(orig_idx)

    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gts]
    return tp_indices, fp_indices, fn_indices


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

        results = model.predict(img, imgsz=imgsz, conf=collection_conf, verbose=False, device=device)[0]
        pred_boxes = []
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