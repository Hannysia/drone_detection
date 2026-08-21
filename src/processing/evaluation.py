import logging
from pathlib import Path
import pandas as pd
import tqdm
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.config import OUTPUT_DIR
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



def evaluate_model(weights_path: Path, yaml_path: Path, run_name: str) -> None:
    """Evaluates the trained model on the 'test' split and prints metrics."""
    log.info("\n" + "="*80)
    log.info(f"STARTING YOLO EVALUATION ON TEST DATA: {run_name.upper()}")
    log.info("="*80)
    log.info(f"Loading weights from: {weights_path.resolve()}")
    
    model = YOLO(str(weights_path))
    
    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=1280,
        batch=8,
        device=0,
        rect=True,
        project=str(OUTPUT_DIR),
        name=f"{run_name}_test_evaluation",
        exist_ok=True,
        save=True,
        plots=True
    )
    
    if metrics is not None and hasattr(metrics, 'results_dict'):
        metrics_dict = metrics.results_dict
        log.info("\n" + "="*50)
        log.info(f"TEST DATA METRICS REPORT ({run_name.upper()})")
        log.info("="*50)
        log.info(f"Precision (P):              {metrics_dict.get('metrics/precision(B)', 0.0):.4f}")
        log.info(f"Recall (R):                 {metrics_dict.get('metrics/recall(B)', 0.0):.4f}")
        log.info(f"mAP50:                      {metrics_dict.get('metrics/mAP50(B)', 0.0):.4f}")
        log.info(f"mAP50-95:                   {metrics_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
        log.info("="*50)


def load_eval_model(weights_path: str, device: torch.device) -> tuple[YOLO, torch.nn.Module]:
    """Loads a trained YOLO model and prepares its PyTorch module for inference."""
    model = YOLO(weights_path)
    pytorch_model = model.model
    pytorch_model.eval()
    pytorch_model.to(device)
    log.info(f"[EVAL] Model loaded from: {weights_path}")
    log.info(f"[EVAL] Parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    return model, pytorch_model


def get_detections_boxes(model: YOLO, img_path: str, conf: float) -> list[tuple[int, int, int, int, float]]:
    """Executes standard YOLO inference to retrieve bounding boxes."""
    results = model(img_path, conf=conf, verbose=False)
    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf_val))
    return boxes


def evaluate_predictions(gt_boxes: list, pred_boxes: list, iou_threshold: float = 0.5) -> tuple[list[int], list[int], list[int]]:
    """Greedy IoU matching between ground truths and predicted bounding boxes."""
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    matched_gts = set()
    tp_indices = []
    fp_indices = []
    
    for pred_idx, pred in enumerate(pred_boxes):
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
            tp_indices.append(pred_idx)
        else:
            fp_indices.append(pred_idx)
            
    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gts]
    return tp_indices, fp_indices, fn_indices


def run_dataset_predictions_db(model: YOLO, test_dir: Path, imgsz: int = 1280) -> list[dict]:
    """Runs batch predictions at conf 0.01 over test split to build evaluation database."""
    test_pos_dir = test_dir / "RAW_POSITIVES"
    test_neg_dir = test_dir / "RAW_NEGATIVES"
    
    test_images = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        if test_pos_dir.exists():
            test_images.extend([(p, True) for p in test_pos_dir.glob(ext)])
        if test_neg_dir.exists():
            test_images.extend([(p, False) for p in test_neg_dir.glob(ext)])
            
    log.info(f"Found {len(test_images)} test images to process.")
    
    raw_database = []
    
    for img_path, is_positive in tqdm(test_images, desc="Running Evaluation Inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]
        
        gt_boxes = []
        if is_positive:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            x_c, y_c, w, h = map(float, parts[1:5])
                            x_min = (x_c - w / 2.0) * w_orig
                            y_min = (y_c - h / 2.0) * h_orig
                            x_max = (x_c + w / 2.0) * w_orig
                            y_max = (y_c + h / 2.0) * h_orig
                            gt_boxes.append([x_min, y_min, x_max, y_max])

        device = 0 if torch.cuda.is_available() else "cpu"
        results = model.predict(img, imgsz=imgsz, verbose=False, device=device)[0]
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
            "pred_boxes": pred_boxes
        })
        
    return raw_database