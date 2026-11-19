import logging
import random
from pathlib import Path
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

from src import config
from src.config import HF_TOKEN, MANIFESTS_DIR, REPO_ID
from src.processing.evaluation import get_detections_boxes, load_eval_model
from src.processing.manifests import select_manifest_frames
from src.processing.yolo_dataset import _validate_run_name
from src.utils.visualization import (
    overlay_heatmap,
    plot_detections_summary,
    plot_eigencam_results,
    plot_false_positive_results,
)

log = logging.getLogger(__name__)

def find_sppf_layer(pytorch_model: torch.nn.Module) -> tuple[torch.nn.Module, int]:
    """Dynamically scans backbones to isolate the SPPF layer."""
    for i, layer in enumerate(pytorch_model.model):
        if "SPPF" in type(layer).__name__:
            log.info(f"[EVAL] SPPF layer dynamically found at index {i}")
            return layer, i
    log.warning("[WARNING] SPPF layer not found automatically. Using fallback index 9")
    return pytorch_model.model[9], 9



def manual_eigencam(
    pytorch_model: torch.nn.Module, 
    target_layer: torch.nn.Module, 
    img_tensor: torch.Tensor
) -> np.ndarray:
    """Computes normalized EigenCAM heatmap via a forward hook and SVD."""
    activations = {}
    def hook(model, input_hook, output_hook):
        activations["feat"] = output_hook.detach()

    handle = target_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = pytorch_model(img_tensor)
    finally:
        handle.remove()

    feat = activations["feat"].float().cpu().squeeze(0).numpy()
    reshaped = feat.reshape(feat.shape[0], -1)
    _, _, Vt = np.linalg.svd(reshaped, full_matrices=False)
    
    cam_map = Vt[0].reshape(feat.shape[1], feat.shape[2])
    if cam_map.mean() < 0:
        cam_map = -cam_map

    cam_map = np.maximum(cam_map, 0)
    cam_map -= cam_map.min()
    if cam_map.max() > 0:
        cam_map /= cam_map.max()
    return cam_map


def letterbox_image(
    img_rgb: np.ndarray,
    imgsz: int = 1280,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resizes an RGB image with square letterbox padding matching YOLO rect=False."""
    if imgsz <= 0 or imgsz % 32 != 0:
        raise ValueError(f"imgsz must be a positive multiple of 32, got {imgsz}")

    h, w = img_rgb.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = (
        cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if (w, h) != (new_w, new_h)
        else img_rgb.copy()
    )

    dw = imgsz - new_w
    dh = imgsz - new_h
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, (top, top + new_h, left, left + new_w)


def preprocess_image_tensor(
    img_path: str | Path,
    device: torch.device,
    imgsz: int = 1280,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Reads an image, applies square letterbox preprocessing, and returns its PyTorch tensor."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Unreadable diagnostic image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_letterboxed, _ = letterbox_image(img_rgb, imgsz=imgsz)
    tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    return img_rgb, img_letterboxed, tensor


def _compute_eigencam_overlay(
    pytorch_model: torch.nn.Module,
    target_layer: torch.nn.Module,
    img_path: str | Path,
    device: torch.device,
    imgsz: int,
) -> tuple[np.ndarray, np.ndarray]:
    img_rgb, img_letterboxed, tensor = preprocess_image_tensor(img_path, device, imgsz=imgsz)
    _, crop_box = letterbox_image(img_rgb, imgsz=imgsz)
    cam_map = manual_eigencam(pytorch_model, target_layer, tensor)
    overlay = overlay_heatmap(
        img_rgb,
        cam_map,
        crop_box=crop_box,
        input_size=img_letterboxed.shape[:2],
    )
    return img_rgb, overlay


def _draw_detection_boxes(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float, float]],
    label_prefix: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    annotated = image.copy()
    for x1, y1, x2, y2, conf_val in boxes:
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), color, 2)
        cv2.putText(
            annotated,
            f"{label_prefix} {float(conf_val):.2f}",
            (ix1, max(iy1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return annotated


def analyze_single_image(
    model: YOLO,
    pytorch_model: torch.nn.Module,
    target_layer: torch.nn.Module,
    img_path: str | Path,
    device: torch.device,
    save_path: Path,
    conf_thresh: float,
    *,
    imgsz: int = 1280,
    show: bool = True,
) -> int:
    """Renders visual layout displaying predictions side-by-side with CAM mappings."""
    img_rgb, overlay = _compute_eigencam_overlay(
        pytorch_model,
        target_layer,
        img_path,
        device,
        imgsz,
    )
    boxes = get_detections_boxes(model, str(img_path), conf=conf_thresh, imgsz=imgsz)
    overlay_with_boxes = _draw_detection_boxes(overlay, boxes, "drone", (0, 255, 0))

    plot_eigencam_results(
        img_rgb=img_rgb,
        overlay=overlay,
        overlay_with_boxes=overlay_with_boxes,
        save_path=save_path,
        filename=Path(img_path).name,
        boxes_count=len(boxes),
        show=show,
    )
    return len(boxes)


def plot_false_positive_inline(
    fp: dict,
    pytorch_model: torch.nn.Module,
    target_layer: torch.nn.Module,
    device: torch.device,
    imgsz: int = 1280,
    save_path: Path | None = None,
    *,
    show: bool = True,
) -> None:
    """Generates EigenCAM attention overlay and plots a False Positive side-by-side."""
    fp_path = Path(fp["path"])
    img_rgb, overlay = _compute_eigencam_overlay(
        pytorch_model,
        target_layer,
        fp_path,
        device,
        imgsz,
    )
    boxes = [
        (float(x1), float(y1), float(x2), float(y2), float(conf))
        for (x1, y1, x2, y2), conf in zip(fp["boxes"], fp["confs"])
    ]
    overlay_boxes = _draw_detection_boxes(overlay, boxes, "FP", (255, 0, 0))
    plot_false_positive_results(
        img_rgb,
        overlay,
        overlay_boxes,
        fp_path.name,
        fp["n_dets"],
        fp["max_conf"],
        save_path=save_path,
        show=show,
    )


def _resolve_weights_path(weights_path: Path, run_name: str) -> Path:
    if weights_path.is_file():
        return weights_path

    hf_file_name = f"models/yolo26n_{run_name}_best.pt"
    if REPO_ID and HF_TOKEN:
        try:
            downloaded = Path(hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_file_name,
                repo_type="dataset",
                token=HF_TOKEN,
            ))
            if downloaded.is_file():
                return downloaded
        except Exception as exc:
            log.warning("Could not fetch %s: %s", hf_file_name, exc)

    fallback = config.OUTPUT_DIR / run_name / "weights" / "best.pt"
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(f"No checkpoint found for run {run_name!r}")


def run_shortcut_analysis(
    weights_path: str | Path,
    dataset_root: str | Path,
    output_dir_path: str | Path,
    run_name: str,
    n_images: int = 15,
    conf_thresh: float = 0.25,
    *,
    split: str = "val",
    category: str = "positive",
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    imgsz: int = 1280,
    show: bool = True,
) -> None:
    """Executes model diagnostics by projecting CAM maps onto manifest-selected frames."""
    validated_name = _validate_run_name(run_name)
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")

    selected_rows = select_manifest_frames(
        Path(dataset_root),
        split=split,
        category=category,
        manifest_path=manifest_path,
        source_path=source_path,
    )

    log.info("\n" + "=" * 80)
    log.info("STARTING SHORTCUT ANALYSIS FOR: %s", validated_name.upper())
    log.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir_path) / validated_name
    output_dir.mkdir(exist_ok=True, parents=True)

    resolved_weights_path = _resolve_weights_path(Path(weights_path), validated_name)
    model, pytorch_model = load_eval_model(str(resolved_weights_path), device)
    target_layer, _ = find_sppf_layer(pytorch_model)

    rng = random.Random(42)
    candidates = [path for path, _ in selected_rows]
    selected_images = rng.sample(candidates, min(n_images, len(candidates)))
    log.info("[INFO] Sampling %d frames for analysis from %s/%s.", len(selected_images), split, category)

    detection_counts = []
    for i, img_path in enumerate(tqdm(selected_images, desc="Generating Visualizations")):
        save_file_path = output_dir / f"cam_{i:03d}_{img_path.stem}.png"
        n_dets = analyze_single_image(
            model,
            pytorch_model,
            target_layer,
            img_path,
            device,
            save_file_path,
            conf_thresh,
            imgsz=imgsz,
            show=show,
        )
        detection_counts.append(n_dets)

    summary_plot_path = output_dir / "detection_summary.png"
    plot_detections_summary(detection_counts, validated_name, summary_plot_path, show=show)

    log.info("\n" + "=" * 50)
    log.info("ANALYSIS REPORT: %s", validated_name.upper())
    log.info("=" * 50)
    log.info("Frames Analyzed:            %d", len(selected_images))
    log.info("Average Detections/Frame:   %.2f", float(np.mean(detection_counts)))
    log.info("Missed Objects:             %d", sum(1 for d in detection_counts if d == 0))
    log.info("Artifacts Saved To:         %s", output_dir.resolve())
    log.info("=" * 50)


def run_false_positive_inference(
    model: YOLO,
    dataset_root: Path,
    conf_thresh: float = 0.25,
    *,
    split: str = "val",
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    imgsz: int = 1280,
    allowed_splits: tuple[str, ...] = ("val", "test"),
) -> list[dict]:
    """Runs inference on manifest-selected negative images to find and sort false positives."""
    all_images = [
        path
        for path, _ in select_manifest_frames(
            Path(dataset_root),
            split=split,
            category="negative",
            manifest_path=manifest_path,
            source_path=source_path,
            allowed_splits=allowed_splits,
        )
    ]
    log.info("Total manifest negatives (%s): %d", split, len(all_images))
    log.info("Running inference with imgsz=%d, conf=%s...", imgsz, conf_thresh)

    device = 0 if torch.cuda.is_available() else "cpu"
    false_positives = []

    for img_path in tqdm(all_images, desc="Running Inference on Negatives"):
        results = model(str(img_path), imgsz=imgsz, conf=conf_thresh, device=device, verbose=False)

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

    log.info("\nFalse positives found: %d / %d", len(false_positives), len(all_images))
    if false_positives:
        log.info("  Max confidence: %.3f", false_positives[0]["max_conf"])
        log.info("  Min confidence: %.3f", false_positives[-1]["max_conf"])

    return false_positives