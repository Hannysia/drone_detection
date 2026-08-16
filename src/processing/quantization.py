import logging
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import HfApi, hf_hub_download

from src.config import REPO_ID, HF_TOKEN, OUTPUT_DIR

log = logging.getLogger(__name__)

def get_file_size_mb(path: Path) -> float:
    """Returns file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)

def benchmark_model(model: YOLO, label: str, dummy_img: np.ndarray, n_warmup: int = 10, n_benchmark: int = 100) -> dict:
    """Measures FPS and latency of the model on a dummy image and prints output."""
    print(f"\nBenchmark: {label}")

    # GPU Warmup
    for _ in range(n_warmup):
        model(dummy_img, verbose=False, device="cuda")
    torch.cuda.synchronize()

    # Latency/FPS measurement
    times = []
    for _ in range(n_benchmark):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(dummy_img, verbose=False, device="cuda")
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # Convert to milliseconds
    fps = 1000 / np.mean(times)

    result = {
        "label": label,
        "mean_ms": np.mean(times),
        "fps": fps,
    }

    print(f"  FPS:         {fps:.1f}")
    print(f"  Latency:     {np.mean(times):.2f} ± {np.std(times):.2f} ms")

    return result

def validate_model_accuracy(model: YOLO, label: str, dataset_yaml: Path, imgsz: int = 1280) -> float:
    """Validates the model on the test split and returns mAP@50 (rect=False for TensorRT)."""
    print(f"\nValidating accuracy for: {label}...")
    
    metrics = model.val(
        data=str(dataset_yaml), 
        split="test", 
        device="cuda", 
        rect=False,
        imgsz=imgsz,
        verbose=False
    )
    
    map50 = metrics.box.map50
    print(f"  [ACCURACY] mAP@50 for {label}: {map50:.4f}")
    return map50

def run_export_and_benchmark_pipeline(
    weights_path: Path, 
    dataset_yaml: Path,
    model_file_name: str,
    imgsz: int = 1280, 
    n_warmup: int = 10, 
    n_benchmark: int = 100
) -> None:
    """Exports model to TensorRT FP16, runs full benchmarks, and uploads .engine checkpoint to HF."""
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    results = []
    sizes = {}
    accuracies = {}
    trt_fp16_path = None

    # -------------------------------------------------
    # 1. Baseline PyTorch Model (FP32)
    # -------------------------------------------------
    print("\n" + "="*50)
    print("1. FP32 PyTorch (baseline)")
    print("="*50)
    model_fp32 = YOLO(str(weights_path)).to("cuda")
    sizes["FP32 PyTorch"] = get_file_size_mb(weights_path)
    print(f"  Size: {sizes['FP32 PyTorch']:.1f} MB")
    
    results.append(benchmark_model(model_fp32, "FP32 PyTorch", dummy_img, n_warmup, n_benchmark))
    accuracies["FP32 PyTorch"] = validate_model_accuracy(model_fp32, "FP32 PyTorch", dataset_yaml, imgsz)

    # -------------------------------------------------
    # 2. TensorRT FP16 (Standard Native Export)
    # -------------------------------------------------
    print("\n" + "="*50)
    print("2. TensorRT FP16 Export")
    print("="*50)
    try:
        print("  Exporting PyTorch to TensorRT FP16...")
        engine_path_str = model_fp32.export(
            format="engine",
            imgsz=imgsz,
            half=True,       
            workspace=4,
            dynamic=False
        )
        
        trt_fp16_path = Path(engine_path_str)
        sizes["TRT FP16"] = get_file_size_mb(trt_fp16_path)
        print(f"  Size: {sizes['TRT FP16']:.1f} MB")
        
        model_trt_fp16 = YOLO(str(trt_fp16_path), task="detect")
        results.append(benchmark_model(model_trt_fp16, "TRT FP16", dummy_img, n_warmup, n_benchmark))
        accuracies["TRT FP16"] = validate_model_accuracy(model_trt_fp16, "TRT FP16", dataset_yaml, imgsz)

    except Exception as e:
        print(f"  TRT FP16 export failed: {e}")

    # -------------------------------------------------
    # 3. Summary Table
    # -------------------------------------------------
    print("\n" + "="*80)
    print("SUMMARY TABLE OF EMPIRICAL RESULTS (KAGGLE TESLA T4)")
    print("="*80)
    print(f"{'Format':<18} {'Size (MB)':<12} {'FPS':<10} {'Latency (ms)':<14} {'mAP@50':<10}")
    print("-" * 80)

    for r in results:
        label = r["label"]
        size = sizes.get(label, 0)
        map50 = accuracies.get(label, 0.0)
        print(f"{label:<18} {size:<12.1f} {r['fps']:<10.1f} {r['mean_ms']:<14.2f} {map50:<10.4f}")

    # -------------------------------------------------
    # 4. HF Upload
    # -------------------------------------------------
    print("\n" + "=" * 50)
    print("4. Publication on Hugging Face")
    print("=" * 50)
    try:
        if trt_fp16_path and trt_fp16_path.exists():
            print("  Uploading .engine...")
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=str(trt_fp16_path),
                path_in_repo=f"models/{trt_fp16_path.name}",
                repo_id=REPO_ID, 
                token=HF_TOKEN, 
                repo_type="dataset"
            )
            print("  ✅ Successfully saved to HF!")
        else:
            print("  ❌ Engine file not found. Skipping HF upload.")
    except Exception as e:
        print(f"  ❌ Upload to HF failed: {e}")