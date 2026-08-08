import logging
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import HfApi
import random
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from src.utils.visualization import plot_detections_summary
from src.config import (
    EXTRACTED_DATA_DIR, OUTPUT_DIR, REPO_ID, HF_TOKEN, WORKING_DIR
)

log = logging.getLogger(__name__)

def safe_symlink(src: Path, dst: Path) -> None:
    """Creates a symbolic link, safely unlinking any existing file or link at destination first."""
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except OSError:
            return
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass

def prepare_yolo_dataset(IMG_DATASET_PATH: Path, working_dir: Path, run_name: str, use_negatives: bool) -> Path:
    """Prepares isolated directories and creates relative symlinks for images and labels."""
    log.info(f"\n[DATA PREPARATION] Setting up dataset directories for run: {run_name}")
    
    run_data_dir = working_dir / f"yolo_dataset_{run_name}"
    
    for split in ["train", "val", "test"]:
        (run_data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (run_data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    def process_split(split_name: str) -> None:
        pos_dir = EXTRACTED_DATA_DIR/ split_name / "RAW_POSITIVES"
        linked_pos_count = 0
        if pos_dir.exists():
            for jpg_path in pos_dir.glob("*.jpg"):
                txt_path = jpg_path.with_suffix(".txt")
                if txt_path.exists():
                    dest_img = run_data_dir / "images" / split_name / jpg_path.name
                    dest_lbl = run_data_dir / "labels" / split_name / txt_path.name
                    
                    safe_symlink(jpg_path, dest_img)
                    safe_symlink(txt_path, dest_lbl)
                    linked_pos_count += 1

        linked_neg_count = 0
        if use_negatives:
            neg_dir = EXTRACTED_DATA_DIR/ split_name / "RAW_NEGATIVES"
            if neg_dir.exists():
                for jpg_path in neg_dir.glob("*.jpg"):
                    dest_img = run_data_dir / "images" / split_name / jpg_path.name
                    safe_symlink(jpg_path, dest_img)
                    linked_neg_count += 1

        log.info(f"[{split_name.upper()}] Symlinked {linked_pos_count:,} positives and {linked_neg_count:,} negatives.")

    process_split("train")
    process_split("val")
    process_split("test")

    yaml_data = {
        "path": str(run_data_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "drone"}
    }
    
    yaml_path = working_dir / f"dataset_{run_name}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    log.info(f"[DATA PREPARATION] Created YAML configuration at: {yaml_path}")
    return yaml_path


def find_run_best_weights(run_name: str) -> Path:
    """Dynamically locates the best weights file for a specific execution run."""
    search_pattern = f"{run_name}*/weights/best.pt"
    found_weights = sorted(list(OUTPUT_DIR.glob(search_pattern)))
    
    if found_weights:
        best_weights = found_weights[-1]
        log.info(f"[INFO] Dynamically detected best weights at: {best_weights.resolve()}")
        return best_weights
    else:
        best_weights = OUTPUT_DIR / run_name / "weights" / "best.pt"
        log.warning(f"[WARNING] Dynamic search failed. Defaulting to: {best_weights.resolve()}")
        return best_weights

def upload_checkpoint_to_hf(weights_path: Path, run_name: str) -> None:
    """Uploads model weights to Hugging Face repository under an isolated path."""
    if REPO_ID and HF_TOKEN and weights_path.exists():
        log.info(f"[HF UPLOAD] Uploading checkpoint for {run_name} to Hugging Face...")
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=str(weights_path),
                path_in_repo=f"models/yolo26n_{run_name}_best.pt",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            log.info("[SUCCESS] Checkpoint successfully uploaded to Hugging Face.")
        except Exception as e:
            log.error(f"[WARNING] Hugging Face upload failed: {e}")
    else:
        log.info("[INFO] Hugging Face upload skipped: credentials or weights file missing.")


def run_yolo_training(
    yaml_path: Path, 
    run_name: str,
    model_type: str = "yolo26n.pt",
    **train_kwargs
) -> Path:
    """
    Initializes and runs the training pipeline.
    """
    log.info("\n" + "="*80)
    log.info(f"STARTING YOLO TRAINING RUN: {run_name.upper()}")
    log.info("="*80)
    log.info(f"Loading weights: {model_type}")

    model = YOLO(model_type)

    default_args = {
        "epochs": 50,
        "imgsz": 1280,
        "batch": 8,
        "patience": 15,
        "device": 0,
        "optimizer": "auto",
        "rect": True,
        "cache": "disk",
        "workers": 2,
        "seed": 42,
        "deterministic": True,
        "project": str(OUTPUT_DIR),
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "plots": True,
        
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "translate": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }

    default_args.update(train_kwargs)

    results = model.train(
        data=str(yaml_path),
        **default_args
    )

    best_weights = find_run_best_weights(run_name)

    if results is not None and hasattr(results, "results_dict"):
        metrics = results.results_dict
        log.info("\n" + "="*50)
        log.info(f"METRICS REPORT ({run_name.upper()})")
        log.info("="*50)
        log.info(f"Precision (P):   {metrics.get('metrics/precision(B)', 0.0):.4f}")
        log.info(f"Recall (R):      {metrics.get('metrics/recall(B)', 0.0):.4f}")
        log.info(f"mAP50:           {metrics.get('metrics/mAP50(B)', 0.0):.4f}")
        log.info(f"mAP50-95:        {metrics.get('metrics/mAP50-95(B)', 0.0):.4f}")
        log.info("="*50)

    upload_checkpoint_to_hf(best_weights, run_name)
    return best_weights



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



import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from src.utils.visualization import overlay_heatmap, plot_eigencam_results


def load_eval_model(weights_path: str, device: torch.device) -> tuple[YOLO, torch.nn.Module]:
    """Loads a trained YOLO model and prepares its PyTorch module for inference."""
    model = YOLO(weights_path)
    pytorch_model = model.model
    pytorch_model.eval()
    pytorch_model.to(device)
    log.info(f"[EVAL] Model loaded from: {weights_path}")
    log.info(f"[EVAL] Parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    return model, pytorch_model


def find_sppf_layer(pytorch_model: torch.nn.Module) -> tuple[torch.nn.Module, int]:
    """Dynamically scans backbones to isolate the SPPF layer."""
    for i, layer in enumerate(pytorch_model.model):
        if "SPPF" in type(layer).__name__:
            log.info(f"[EVAL] SPPF layer dynamically found at index {i}")
            return layer, i
    log.warning("[WARNING] SPPF layer not found automatically. Using fallback index 9")
    return pytorch_model.model[9], 9


def run_eigencam_gradcam(
    pytorch_model: torch.nn.Module, 
    target_layer: torch.nn.Module, 
    img_tensor: torch.Tensor
) -> np.ndarray | None:
    """EigenCAM implementation relying on pytorch-grad-cam library wrapper."""
    try:
        from pytorch_grad_cam import EigenCAM

        class YOLOWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                output = self.model(x)
                if isinstance(output, (list, tuple)):
                    preds = output[0] if isinstance(output[0], torch.Tensor) else output
                    if preds.dim() == 3:
                        return preds[:, :, 4:5].squeeze(-1)
                    return preds
                return output

        wrapped = YOLOWrapper(pytorch_model)
        cam = EigenCAM(model=wrapped, target_layers=[target_layer])
        with torch.no_grad():
            grayscale_cam = cam(input_tensor=img_tensor)
        return grayscale_cam[0]
    except Exception:
        return None


def manual_eigencam(
    pytorch_model: torch.nn.Module, 
    target_layer: torch.nn.Module, 
    img_tensor: torch.Tensor
) -> np.ndarray:
    """Manual SVD-based EigenCAM fallback when third-party wrappers fail."""
    activations = {}
    def hook(model, input_hook, output_hook):
        activations["feat"] = output_hook.detach()

    handle = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = pytorch_model(img_tensor)
    handle.remove()

    feat = activations["feat"].cpu().squeeze(0).numpy()  # Shape: (C, H, W)
    reshaped = feat.reshape(feat.shape[0], -1)           # Shape: (C, H*W)
    _, _, Vt = np.linalg.svd(reshaped, full_matrices=False)
    
    cam_map = Vt[0].reshape(feat.shape[1], feat.shape[2])
    if cam_map.mean() < 0:
        cam_map = -cam_map

    cam_map = np.maximum(cam_map, 0)
    cam_map -= cam_map.min()
    if cam_map.max() > 0:
        cam_map /= cam_map.max()
    return cam_map


def preprocess_image_tensor(img_path: str, device: torch.device) -> tuple[np.ndarray | None, np.ndarray | None, torch.Tensor | None]:
    """Rescales raw validation frames and constructs PyTorch tensors."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    return img_rgb, img_resized, tensor


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


def analyze_single_image(
    model: YOLO, 
    pytorch_model: torch.nn.Module, 
    target_layer: torch.nn.Module, 
    img_path: str, 
    device: torch.device, 
    save_path: Path, 
    conf_thresh: float
) -> int | None:
    """Renders visual layout displaying predictions side-by-side with CAM mappings."""
    img_rgb, _, tensor = preprocess_image_tensor(img_path, device)
    if img_rgb is None:
        return None

    cam_map = run_eigencam_gradcam(pytorch_model, target_layer, tensor)
    if cam_map is None:
        cam_map = manual_eigencam(pytorch_model, target_layer, tensor)

    overlay = overlay_heatmap(img_rgb, cam_map)
    boxes = get_detections_boxes(model, img_path, conf=conf_thresh)

    overlay_with_boxes = overlay.copy()
    for (x1, y1, x2, y2, conf_val) in boxes:
        cv2.rectangle(overlay_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay_with_boxes, f"drone {conf_val:.2f}",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    plot_eigencam_results(
        img_rgb=img_rgb,
        overlay=overlay,
        overlay_with_boxes=overlay_with_boxes,
        save_path=save_path,
        filename=Path(img_path).name,
        boxes_count=len(boxes)
    )

    return len(boxes)



def run_full_pipeline(use_negatives: bool, run_name: str, **train_kwargs) -> Path:
    """Executes the setup, training, and subsequent test evaluation pipeline with dynamic parameters."""
    if not EXTRACTED_DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset path {EXTRACTED_DATA_DIR} does not exist.")
        
    yaml_path = prepare_yolo_dataset(EXTRACTED_DATA_DIR, WORKING_DIR, run_name, use_negatives)
    best_weights = run_yolo_training(yaml_path, run_name, **train_kwargs)
    evaluate_model(best_weights, yaml_path, run_name)
    
    return best_weights


def run_shortcut_analysis(
    weights_path: str, 
    val_dir: str, 
    output_dir_path: str, 
    run_name: str, 
    n_images: int = 15, 
    conf_thresh: float = 0.25
) -> None:
    """Executes model diagnostics by projecting CAM maps onto sampled verification frames."""
    log.info("\n" + "="*80)
    log.info(f"STARTING SHORTCUT ANALYSIS FOR: {run_name.upper()}")
    log.info("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir_path) / run_name
    output_dir.mkdir(exist_ok=True, parents=True)

    resolved_weights_path = None
    hf_file_name = f"models/yolo26n_{run_name}_best.pt"

    if REPO_ID and HF_TOKEN:
        log.info(f"[INFO] Attempting download from HF Hub: {hf_file_name}")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_file_name,
                repo_type="dataset",
                token=HF_TOKEN
            )
            resolved_weights_path = Path(downloaded)
            log.info(f"[SUCCESS] Weights downloaded to secure cache: {resolved_weights_path}")
        except Exception as e:
            log.warning(f"[WARNING] Could not fetch from Hugging Face Hub: {e}")

    if resolved_weights_path is None or not resolved_weights_path.exists():
        direct_path = Path(weights_path)
        if direct_path.exists():
            resolved_weights_path = direct_path
        else:
            log.info("[INFO] Fallback searching local directories...")
            search_pattern = f"runs/detect/yolo26_shortcut_analysis/{run_name}*/weights/best.pt"
            found_weights = sorted(list(WORKING_DIR.glob(search_pattern)))
            if found_weights:
                resolved_weights_path = found_weights[-1]
                log.info(f"[SUCCESS] Local checkpoint located: {resolved_weights_path}")
            else:
                log.error("❌ [ERROR] Could not resolve any valid 'best.pt' file.")
                return

    model, pytorch_model = load_eval_model(str(resolved_weights_path), device)
    target_layer, _ = find_sppf_layer(pytorch_model)

    all_images = sorted(list(Path(val_dir).glob("*.jpg")))
    if not all_images:
        log.error(f"❌ [ERROR] Verification folder is empty: {val_dir}")
        return

    random.seed(42)
    selected_images = random.sample(all_images, min(n_images, len(all_images)))
    log.info(f"[INFO] Sampling {len(selected_images)} frames for analysis.")

    detection_counts = []
    for i, img_path in enumerate(tqdm(selected_images, desc="Generating Visualizations")):
        save_file_path = output_dir / f"cam_{i:03d}_{img_path.stem}.png"
        n_dets = analyze_single_image(
            model, pytorch_model, target_layer,
            str(img_path), device, save_file_path, conf_thresh
        )
        if n_dets is not None:
            detection_counts.append(n_dets)

    plot_detections_summary(detection_counts, run_name)

    log.info("\n" + "="*50)
    log.info(f"ANALYSIS REPORT: {run_name.upper()}")
    log.info("="*50)
    log.info(f"Frames Analyzed:            {len(selected_images)}")
    log.info(f"Average Detections/Frame:   {np.mean(detection_counts):.2f}")
    log.info(f"Missed Objects:             {sum(1 for d in detection_counts if d == 0)}")
    log.info(f"Artifacts Saved To:         {output_dir.resolve()}")
    log.info("="*50)



def run_false_positive_inference(model: YOLO, negatives_dir: Path, conf_thresh: float) -> list[dict]:
    """Runs inference on all negative images to find and sort false positives."""
    all_images = sorted(list(negatives_dir.glob("*.jpg")))
    log.info(f"Total negatives found: {len(all_images)}")
    log.info(f"Running inference with confidence threshold={conf_thresh}...")

    false_positives = []
    
    for img_path in tqdm(all_images, desc="Running Inference on Negatives"):
        results = model(str(img_path), conf=conf_thresh, verbose=False)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            max_conf = float(boxes.conf.max())

            false_positives.append({
                "path": img_path,
                "n_dets": len(boxes),
                "max_conf": max_conf,
                "boxes": boxes.xyxy.cpu().numpy(),
                "confs": boxes.conf.cpu().numpy(),
            })

    false_positives.sort(key=lambda x: x["max_conf"], reverse=True)

    log.info(f"\nFalse positives found: {len(false_positives)} / {len(all_images)}")
    if false_positives:
        log.info(f"  Max confidence: {false_positives[0]['max_conf']:.3f}")
        log.info(f"  Min confidence: {false_positives[-1]['max_conf']:.3f}")

    return false_positives