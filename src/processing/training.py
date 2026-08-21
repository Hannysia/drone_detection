import logging
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import HfApi
from src.config import (
    OUTPUT_DIR, REPO_ID, HF_TOKEN, WORKING_DIR, EXTRACTED_DATA_DIR
)
from src.processing.dataset import prepare_yolo_dataset

log = logging.getLogger(__name__)

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
    """Initializes and runs the training pipeline with custom train kwargs."""
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
    results = model.train(data=str(yaml_path), **default_args)

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

def run_full_pipeline(use_negatives: bool, run_name: str, **train_kwargs) -> Path:
    """Executes the setup, training, and subsequent test evaluation pipeline with dynamic parameters."""
    from src.processing.evaluation import evaluate_model
    
    if not EXTRACTED_DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset path {EXTRACTED_DATA_DIR} does not exist.")
        
    yaml_path = prepare_yolo_dataset(EXTRACTED_DATA_DIR, WORKING_DIR, run_name, use_negatives)
    best_weights = run_yolo_training(yaml_path, run_name, **train_kwargs)
    evaluate_model(best_weights, yaml_path, run_name)
    
    return best_weights