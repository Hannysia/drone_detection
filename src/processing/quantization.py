from collections import Counter, deque
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import math
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import platform
import queue
import random
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from huggingface_hub import HfApi, constants as hf_constants, hf_hub_download
import numpy as np
import onnx
import pandas as pd
import torch
from ultralytics import YOLO
import yaml

from src import config
from src.config import (
    HF_MODEL_PATH_PT,
    HF_TOKEN,
    MANIFESTS_DIR,
    QUANTIZATION_RESOLUTIONS,
    REPO_ID,
    YOLO_MODEL_STEM,
    get_quantized_model_paths,
)
from src.processing.dataset import parse_frame_filename_logic
from src.processing.diagnostics import letterbox_image
from src.processing.evaluation import (
    _collect_matched_detections,
    _compute_average_precision,
    evaluate_predictions,
    evaluate_threshold_free_slices,
    run_dataset_predictions_db,
)
from src.processing.manifests import select_manifest_frames
from src.processing.training import assert_hf_ready
from src.processing.yolo_dataset import (
    _remove_existing_path,
    _validate_run_name,
    cleanup_yolo_dataset,
)
from src.utils.io import parse_frame_name, parse_video_id

log = logging.getLogger(__name__)

DEFAULT_CALIBRATION_IMAGES = 300
MAX_ONNX_IR_VERSION = 10
HF_UPLOAD_TIMEOUT_SECONDS = 180.0
ALL_STAGES = (
    "pytorch_fp32",
    "onnx_fp32",
    "onnx_int8",
    "openvino_fp32",
    "openvino_int8",
)

_HEAD_PREFIX_RE = re.compile(r"(/?model\.(\d+))(?=/|\.)")
_POSTPROCESS_OP_TYPES = frozenset({"TopK", "Mod", "NonMaxSuppression"})
_SENSITIVE_HEAD_OP_TYPES = frozenset({
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Sigmoid",
    "Softmax",
    "TopK",
    "Mod",
    "Gather",
    "GatherElements",
    "NonMaxSuppression",
    "ReduceMax",
    "Cast",
    "Concat",
    "Reshape",
    "Split",
    "Transpose",
    "Slice",
    "Range",
    "Where",
    "Shape",
    "ConstantOfShape",
    "Expand",
    "Unsqueeze",
    "Squeeze",
})
_FALLBACK_SENSITIVE_OP_TYPES = frozenset({
    "Sigmoid",
    "Softmax",
    "TopK",
    "Mod",
    "Gather",
    "GatherElements",
    "NonMaxSuppression",
})


def verify_hf_access(
    repo_id: str | None = None,
    token: str | None = None,
) -> tuple[str, str]:
    """Validates Hugging Face write access and dataset repository visibility."""
    return assert_hf_ready(repo_id=repo_id, token=token)


def resolve_source_weights(
    weights_path: Path | None = None,
    *,
    cache_dir: Path | None = None,
    repo_id: str | None = None,
    token: str | None = None,
) -> Path:
    """Returns a verified local .pt checkpoint or downloads baseline weights from HF."""
    if weights_path is not None:
        resolved = Path(weights_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Source PyTorch weights not found: {resolved}")
        return resolved

    if cache_dir is not None:
        cached = Path(cache_dir) / HF_MODEL_PATH_PT
        if cached.is_file():
            return cached

    target_repo = REPO_ID if repo_id is None else repo_id
    target_token = HF_TOKEN if token is None else token
    if not isinstance(target_repo, str) or not target_repo.strip():
        raise ValueError("REPO_ID is required to download model weights")
    if not isinstance(target_token, str) or not target_token.strip():
        raise ValueError("HF_TOKEN is required to download model weights")

    download_kwargs = {
        "repo_id": target_repo.strip(),
        "filename": HF_MODEL_PATH_PT,
        "repo_type": "dataset",
        "token": target_token.strip(),
    }
    if cache_dir is not None:
        download_kwargs["local_dir"] = str(cache_dir)

    downloaded = Path(hf_hub_download(**download_kwargs))
    if not downloaded.is_file():
        raise FileNotFoundError(f"Downloaded weights missing at: {downloaded}")
    return downloaded


def get_artifact_size_mb(path: Path) -> float:
    """Returns serialized artifact size in megabytes for a file or OpenVINO IR directory."""
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        if not files:
            raise FileNotFoundError(f"Model artifact directory is empty: {path}")
        return sum(item.stat().st_size for item in files) / (1024 * 1024)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    raise FileNotFoundError(f"Model artifact not found: {path}")


def _frame_sort_key(path: Path) -> tuple[int, str]:
    try:
        return parse_frame_name(path)[1], path.name
    except ValueError:
        return 0, path.name


def _interleave_farthest_order(items: list[Path]) -> list[Path]:
    if len(items) <= 2:
        return list(items)
    ordered = []
    queue = deque([(0, len(items) - 1)])
    while queue:
        left, right = queue.popleft()
        mid = (left + right) // 2
        ordered.append(items[mid])
        if left <= mid - 1:
            queue.append((left, mid - 1))
        if mid + 1 <= right:
            queue.append((mid + 1, right))
    return ordered


def _allocate_stratum_quotas(
    stratum_sizes: dict[tuple[str, str, str, str], int],
    max_images: int,
) -> dict[tuple[str, str, str, str], int]:
    total_available = sum(stratum_sizes.values())
    if total_available <= max_images:
        return dict(stratum_sizes)

    ordered_keys = sorted(stratum_sizes)
    if max_images <= len(ordered_keys):
        ranked = sorted(ordered_keys, key=lambda key: (-stratum_sizes[key], key))
        return {key: 1 if idx < max_images else 0 for idx, key in enumerate(ranked)}

    quotas = {key: 1 for key in ordered_keys}
    remaining = max_images - len(ordered_keys)
    surplus_total = sum(stratum_sizes[key] - 1 for key in ordered_keys)
    remainders = []

    for key in ordered_keys:
        surplus = stratum_sizes[key] - 1
        exact_extra = (remaining * surplus) / surplus_total
        extra = min(surplus, math.floor(exact_extra))
        quotas[key] += extra
        remainders.append((exact_extra - extra, key))

    leftover = max_images - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        if quotas[key] < stratum_sizes[key]:
            quotas[key] += 1
            leftover -= 1

    return quotas


def _sample_stratum_round_robin(
    paths: list[Path],
    quota: int,
    rng: random.Random,
) -> list[Path]:
    if quota <= 0:
        return []

    by_video: dict[str, list[Path]] = {}
    for path in paths:
        video_id = parse_video_id(path.name) or path.stem
        by_video.setdefault(video_id, []).append(path)

    video_queues = {
        video_id: deque(_interleave_farthest_order(sorted(frames, key=_frame_sort_key)))
        for video_id, frames in by_video.items()
    }
    video_ids = sorted(video_queues)
    rng.shuffle(video_ids)

    if len(video_ids) >= quota:
        return [video_queues[video_id].popleft() for video_id in video_ids[:quota]]

    selected = []
    active = list(video_ids)
    while len(selected) < quota and active:
        next_active = []
        for video_id in active:
            if len(selected) >= quota:
                break
            queue = video_queues[video_id]
            if queue:
                selected.append(queue.popleft())
            if queue:
                next_active.append(video_id)
        active = next_active
    return selected


def prepare_calibration_dataset(
    dataset_root: Path,
    working_dir: Path,
    *,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    max_images: int = DEFAULT_CALIBRATION_IMAGES,
    seed: int = 42,
    run_name: str = "calibration_ptq",
    overwrite: bool = True,
) -> tuple[Path, list[Path]]:
    """Builds a leak-free train-only stratified calibration dataset shared by ONNX and OpenVINO."""
    if max_images <= 0:
        raise ValueError(f"max_images must be positive, got {max_images}")

    validated_name = _validate_run_name(run_name)
    active_train = select_manifest_frames(
        Path(dataset_root),
        split="train",
        manifest_path=manifest_path,
        source_path=source_path,
        allowed_splits=("train",),
    )

    strata: dict[tuple[str, str, str, str], list[tuple[Path, str]]] = {}
    for img_path, category in active_train:
        meta = parse_frame_filename_logic(img_path.name)
        key = (category, meta["shape"], meta["camera"], meta["background"])
        strata.setdefault(key, []).append((img_path, category))

    quotas = _allocate_stratum_quotas(
        {key: len(items) for key, items in strata.items()},
        max_images,
    )

    rng = random.Random(seed)
    selected_entries: list[tuple[Path, str]] = []
    category_lookup: dict[Path, str] = {}
    for key in sorted(strata):
        items = strata[key]
        for path, category in items:
            category_lookup[path] = category
        sampled_paths = _sample_stratum_round_robin(
            [path for path, _ in items],
            quotas[key],
            rng,
        )
        selected_entries.extend((path, category_lookup[path]) for path in sampled_paths)

    selected_entries.sort(key=lambda entry: entry[0].name)
    working_dir = Path(working_dir)
    output_dir = working_dir / f"yolo_dataset_{validated_name}"
    yaml_path = working_dir / f"dataset_{validated_name}.yaml"

    if not overwrite and (
        output_dir.exists()
        or output_dir.is_symlink()
        or yaml_path.exists()
        or yaml_path.is_symlink()
    ):
        raise FileExistsError(f"Calibration dataset already exists for {validated_name}")

    working_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    with TemporaryDirectory(dir=working_dir) as tmp:
        staged = Path(tmp) / "dataset"
        (staged / "images" / "cal").mkdir(parents=True)
        (staged / "labels" / "cal").mkdir(parents=True)

        for img_path, category in selected_entries:
            files = [("images", img_path)]
            if category == "positive":
                label_path = img_path.with_suffix(".txt")
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing calibration label: {label_path}")
                files.append(("labels", label_path))

            for kind, src in files:
                dest = staged / kind / "cal" / src.name
                try:
                    dest.symlink_to(src)
                except OSError as exc:
                    if getattr(exc, "winerror", None) != 1314:
                        raise
                    shutil.copy2(src, dest)
            counts[category] += 1

        cfg = {
            "path": str(output_dir.resolve()),
            "train": "images/cal",
            "val": "images/cal",
            "names": {0: "drone"},
        }
        staged_yaml = Path(tmp) / "dataset.yaml"
        staged_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        if overwrite:
            cleanup_yolo_dataset(working_dir, validated_name)

        staged.rename(output_dir)
        try:
            os.link(staged_yaml, yaml_path)
        except OSError:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

    staged_images = sorted((output_dir / "images" / "cal").glob("*.jpg"))
    log.info(
        "Prepared train-only calibration set %s: %d images (%d pos, %d neg, %d strata)",
        validated_name,
        len(staged_images),
        counts["positive"],
        counts["negative"],
        len([q for q in quotas.values() if q > 0]),
    )
    return yaml_path, staged_images


def cleanup_calibration_dataset(
    working_dir: Path,
    run_name: str = "calibration_ptq",
) -> None:
    """Removes the temporary calibration dataset directory and YAML config."""
    cleanup_yolo_dataset(Path(working_dir), run_name)


def _move_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        _remove_existing_path(destination)
        shutil.move(str(source), str(destination))
    return destination


def _validate_openvino_dir(model_dir: Path) -> Path:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Exported OpenVINO directory not found: {model_dir}")
    xml_files = sorted(model_dir.glob("*.xml"))
    if not xml_files or not any(xml.with_suffix(".bin").is_file() for xml in xml_files):
        raise FileNotFoundError(f"OpenVINO IR (.xml/.bin) not found in: {model_dir}")
    return model_dir


def export_onnx_fp32(
    pt_path: Path,
    output_path: Path,
    *,
    imgsz: int,
) -> Path:
    """Exports a PyTorch YOLO checkpoint to static CPU ONNX FP32."""
    if not pt_path.is_file():
        raise FileNotFoundError(f"PyTorch weights not found: {pt_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as tmp_dir:
        tmp_pt = Path(tmp_dir) / f"{output_path.stem}.pt"
        shutil.copy2(pt_path, tmp_pt)
        model = YOLO(str(tmp_pt))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=imgsz,
                device="cpu",
                dynamic=False,
                batch=1,
                simplify=True,
                half=False,
            )
        )
        if not exported.is_file():
            raise FileNotFoundError(f"Exported ONNX FP32 file not found: {exported}")
        _move_artifact(exported, output_path)

    if not output_path.is_file():
        raise FileNotFoundError(f"ONNX FP32 output missing after move: {output_path}")
    return output_path


def _extract_detect_head_prefix(model_proto: onnx.ModelProto) -> str | None:
    candidates: dict[str, tuple[int, bool]] = {}
    for node in model_proto.graph.node:
        if not node.name:
            continue
        match = _HEAD_PREFIX_RE.search(node.name)
        if not match:
            continue
        prefix, idx_str = match.group(1), match.group(2)
        lowered = node.name.lower()
        is_head_marker = (
            "one2one" in lowered
            or "/dfl" in lowered
            or ".dfl" in lowered
            or node.op_type in {"TopK", "Mod", "Sigmoid"}
        )
        prev_idx, prev_marker = candidates.get(prefix, (int(idx_str), False))
        candidates[prefix] = (max(prev_idx, int(idx_str)), prev_marker or is_head_marker)

    if not candidates:
        return None
    marked = [prefix for prefix, (_, marker) in candidates.items() if marker]
    pool = marked if marked else list(candidates)
    return max(pool, key=lambda prefix: candidates[prefix][0])


def select_onnx_nodes_to_exclude(
    model_proto: onnx.ModelProto,
    *,
    exclude_head_convs: bool = False,
) -> list[str]:
    """Selects Detect head decode, DFL, Sigmoid, and end2end TopK nodes to keep in FP32."""
    head_prefix = _extract_detect_head_prefix(model_proto)
    excluded = []

    for node in model_proto.graph.node:
        name = node.name
        if not name:
            continue
        lowered = name.lower()
        is_dfl = "/dfl" in lowered or ".dfl" in lowered
        in_head = (
            head_prefix is not None
            and (name == head_prefix or name.startswith(f"{head_prefix}/") or name.startswith(f"{head_prefix}."))
        )

        if node.op_type in _POSTPROCESS_OP_TYPES or is_dfl:
            excluded.append(name)
        elif in_head and (exclude_head_convs or node.op_type in _SENSITIVE_HEAD_OP_TYPES):
            excluded.append(name)
        elif head_prefix is None and (
            node.op_type in _FALLBACK_SENSITIVE_OP_TYPES or "postprocess" in lowered
        ):
            excluded.append(name)

    return list(dict.fromkeys(excluded))


def _preprocess_onnx_image(image_path: Path, imgsz: int) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Unreadable calibration image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded, _ = letterbox_image(rgb, imgsz=imgsz)
    chw = np.transpose(padded, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None, ...])


def _build_calibration_reader(image_paths: list[Path], imgsz: int, input_name: str):
    from onnxruntime.quantization import CalibrationDataReader

    class _StreamingReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._paths = list(image_paths)
            self._index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self._index >= len(self._paths):
                return None
            path = self._paths[self._index]
            self._index += 1
            return {input_name: _preprocess_onnx_image(path, imgsz)}

        def rewind(self) -> None:
            self._index = 0

    return _StreamingReader()


def _preprocess_onnx_for_quantization(source_path: Path, pre_path: Path) -> Path:
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quant_pre_process(
            str(source_path),
            str(pre_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,
            auto_merge=True,
        )
        if pre_path.is_file():
            return pre_path
    except Exception as exc:
        log.warning("quant_pre_process failed (%s); falling back to onnx shape inference.", exc)

    try:
        onnx.shape_inference.infer_shapes_path(str(source_path), str(pre_path))
        if pre_path.is_file():
            return pre_path
    except Exception as exc:
        log.warning("ONNX shape inference fallback failed (%s); using raw FP32 ONNX.", exc)

    shutil.copy2(source_path, pre_path)
    return pre_path


def _copy_onnx_metadata_and_ir(source_model: onnx.ModelProto, target_path: Path) -> None:
    quantized = onnx.load(str(target_path))
    existing = {prop.key: prop.value for prop in quantized.metadata_props}
    for prop in source_model.metadata_props:
        if prop.key not in existing:
            meta = quantized.metadata_props.add()
            meta.key = prop.key
            meta.value = prop.value
    if getattr(quantized, "ir_version", 0) > MAX_ONNX_IR_VERSION:
        quantized.ir_version = MAX_ONNX_IR_VERSION
    onnx.save(quantized, str(target_path))


def _finalize_quantized_onnx(
    source_model: onnx.ModelProto,
    staged_path: Path,
    output_path: Path,
    *,
    ort_module,
) -> Path:
    if not staged_path.is_file():
        raise FileNotFoundError(f"Quantized ONNX INT8 output was not created: {staged_path}")
    _copy_onnx_metadata_and_ir(source_model, staged_path)
    ort_module.InferenceSession(str(staged_path), providers=["CPUExecutionProvider"])
    return _move_artifact(staged_path, output_path)


def quantize_onnx_int8(
    onnx_fp32_path: Path,
    output_path: Path,
    calibration_images: list[Path],
    *,
    imgsz: int,
    calibrate_method: str = "MinMax",
    exclude_head_convs: bool = False,
    moving_average: bool = False,
    calibration_symmetric: bool | None = None,
    percentile: float | None = None,
) -> Path:
    """Quantizes an ONNX FP32 model to QDQ INT8 with end2end head exclusion and ARM64 parity."""
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    if not onnx_fp32_path.is_file():
        raise FileNotFoundError(f"ONNX FP32 model not found: {onnx_fp32_path}")
    if not calibration_images:
        raise ValueError("calibration_images must not be empty")

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    ort_method = method_map.get(calibrate_method.strip().lower())
    if ort_method is None:
        raise ValueError(f"Unsupported ONNX calibrate_method: {calibrate_method!r}")
    if moving_average and ort_method != CalibrationMethod.MinMax:
        raise ValueError("moving_average calibration option is only valid with MinMax")

    source_model = onnx.load(str(onnx_fp32_path))
    if not source_model.graph.input:
        raise ValueError(f"ONNX model has no graph inputs: {onnx_fp32_path}")
    input_name = source_model.graph.input[0].name

    extra_options: dict[str, object] = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }
    if moving_average:
        extra_options["CalibMovingAverage"] = True
        extra_options["CalibMovingAverageConstant"] = 0.01
    if calibration_symmetric is not None:
        extra_options["CalibTensorRangeSymmetric"] = bool(calibration_symmetric)
    if percentile is not None:
        extra_options["CalibPercentile"] = float(percentile)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as tmp_dir:
        tmp_root = Path(tmp_dir)
        pre_path = _preprocess_onnx_for_quantization(onnx_fp32_path, tmp_root / "pre.onnx")
        pre_model = onnx.load(str(pre_path))
        nodes_to_exclude = select_onnx_nodes_to_exclude(
            pre_model,
            exclude_head_convs=exclude_head_convs,
        )
        reader = _build_calibration_reader(calibration_images, imgsz, input_name)

        tmp_int8 = tmp_root / output_path.name
        quantize_static(
            model_input=str(pre_path),
            model_output=str(tmp_int8),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            nodes_to_exclude=nodes_to_exclude,
            calibrate_method=ort_method,
            calibration_providers=["CPUExecutionProvider"],
            extra_options=extra_options,
        )
        _finalize_quantized_onnx(source_model, tmp_int8, output_path, ort_module=ort)

    log.info(
        "Exported ONNX INT8 (%s, moving_average=%s, exclude_head_convs=%s, excluded %d nodes) -> %s",
        calibrate_method,
        moving_average,
        exclude_head_convs,
        len(nodes_to_exclude),
        output_path,
    )
    return output_path


def _quantize_onnx_with_fallback(
    onnx_fp32_path: Path,
    output_path: Path,
    calibration_images: list[Path],
    *,
    imgsz: int,
    calibrate_method: str,
) -> tuple[Path, list[str], bool]:
    try:
        out = quantize_onnx_int8(
            onnx_fp32_path,
            output_path,
            calibration_images,
            imgsz=imgsz,
            calibrate_method=calibrate_method,
            exclude_head_convs=False,
        )
        excluded = select_onnx_nodes_to_exclude(onnx.load(str(onnx_fp32_path)), exclude_head_convs=False)
        return out, excluded, False
    except Exception as exc:
        log.warning("ONNX INT8 failed (%s); retrying with full Detect head excluded.", exc)
        out = quantize_onnx_int8(
            onnx_fp32_path,
            output_path,
            calibration_images,
            imgsz=imgsz,
            calibrate_method=calibrate_method,
            exclude_head_convs=True,
        )
        excluded = select_onnx_nodes_to_exclude(onnx.load(str(onnx_fp32_path)), exclude_head_convs=True)
        return out, excluded, True

def _export_openvino_variant(
    pt_path: Path,
    output_dir: Path,
    *,
    imgsz: int,
    int8: bool,
    calibration_yaml: Path | None = None,
) -> Path:
    if not pt_path.is_file():
        raise FileNotFoundError(f"PyTorch weights not found: {pt_path}")
    if int8 and (calibration_yaml is None or not Path(calibration_yaml).is_file()):
        raise FileNotFoundError(f"Calibration YAML not found for OpenVINO INT8: {calibration_yaml}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp_dir:
        tmp_pt = Path(tmp_dir) / f"{pt_path.stem}.pt"
        shutil.copy2(pt_path, tmp_pt)
        model = YOLO(str(tmp_pt))
        export_kwargs = {
            "format": "openvino",
            "imgsz": imgsz,
            "device": "cpu",
            "dynamic": False,
            "batch": 1,
            "half": False,
            "int8": int8,
        }
        if int8:
            export_kwargs["data"] = str(calibration_yaml)

        exported_dir = Path(model.export(**export_kwargs))
        _validate_openvino_dir(exported_dir)
        _move_artifact(exported_dir, output_dir)

    return _validate_openvino_dir(output_dir)


def export_openvino_fp32(
    pt_path: Path,
    output_dir: Path,
    *,
    imgsz: int,
) -> Path:
    """Exports a PyTorch YOLO checkpoint to static CPU OpenVINO FP32 IR."""
    return _export_openvino_variant(pt_path, output_dir, imgsz=imgsz, int8=False)


def export_openvino_int8(
    pt_path: Path,
    output_dir: Path,
    calibration_yaml: Path,
    *,
    imgsz: int,
) -> Path:
    """Exports and quantizes a PyTorch YOLO checkpoint to OpenVINO INT8 IR via train-only calibration."""
    return _export_openvino_variant(
        pt_path,
        output_dir,
        imgsz=imgsz,
        int8=True,
        calibration_yaml=calibration_yaml,
    )


def quantize_onnx_int8_nncf(
    onnx_fp32_path: Path,
    output_path: Path,
    calibration_images: list[Path],
    *,
    imgsz: int,
    exclude_head_convs: bool = False,
    fast_bias_correction: bool = True,
) -> Path:
    """Quantizes an ONNX FP32 model to INT8 using NNCF with bias correction."""
    import nncf
    import onnxruntime as ort

    if not onnx_fp32_path.is_file():
        raise FileNotFoundError(f"ONNX FP32 model not found: {onnx_fp32_path}")
    if not calibration_images:
        raise ValueError("calibration_images must not be empty")

    source_model = onnx.load(str(onnx_fp32_path))
    if not source_model.graph.input:
        raise ValueError(f"ONNX model has no graph inputs: {onnx_fp32_path}")
    input_name = source_model.graph.input[0].name

    nodes_to_exclude = select_onnx_nodes_to_exclude(
        source_model,
        exclude_head_convs=exclude_head_convs,
    )
    ignored_scope = (
        nncf.IgnoredScope(names=nodes_to_exclude, validate=False)
        if nodes_to_exclude
        else None
    )
    cal_dataset = nncf.Dataset(
        calibration_images,
        lambda path: {input_name: _preprocess_onnx_image(path, imgsz)},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as tmp_dir:
        tmp_int8 = Path(tmp_dir) / output_path.name
        quantized_model = nncf.quantize(
            model=source_model,
            calibration_dataset=cal_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(calibration_images),
            fast_bias_correction=fast_bias_correction,
            ignored_scope=ignored_scope,
        )
        onnx.save(quantized_model, str(tmp_int8))
        _finalize_quantized_onnx(source_model, tmp_int8, output_path, ort_module=ort)

    log.info(
        "Exported ONNX INT8 NNCF (exclude_head_convs=%s, excluded %d nodes) -> %s",
        exclude_head_convs,
        len(nodes_to_exclude),
        output_path,
    )
    return output_path


def export_openvino_int8_nncf(
    openvino_fp32_dir: Path,
    output_dir: Path,
    calibration_images: list[Path],
    *,
    imgsz: int,
    fast_bias_correction: bool = False,
) -> Path:
    """Quantizes an existing OpenVINO FP32 IR model via direct NNCF with Detect head ignored."""
    import nncf
    import openvino as ov

    fp32_dir = _validate_openvino_dir(Path(openvino_fp32_dir))
    if not calibration_images:
        raise ValueError("calibration_images must not be empty")

    xml_src = next(sorted(fp32_dir.glob("*.xml")).__iter__())
    core = ov.Core()
    ov_model = core.read_model(model=str(xml_src), weights=str(xml_src.with_suffix(".bin")))

    cal_dataset = nncf.Dataset(
        calibration_images,
        lambda path: _preprocess_onnx_image(path, imgsz),
    )
    ignored_scope = nncf.IgnoredScope(
        patterns=[r".*model\.23/.*"],
        types=["Sigmoid"],
        validate=False,
    )

    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=cal_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calibration_images),
        fast_bias_correction=fast_bias_correction,
        ignored_scope=ignored_scope,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as tmp_dir:
        tmp_out = Path(tmp_dir) / output_dir.name
        tmp_out.mkdir(parents=True, exist_ok=True)
        ov.save_model(quantized_model, str(tmp_out / xml_src.name), compress_to_fp16=False)
        meta_src = fp32_dir / "metadata.yaml"
        if meta_src.is_file():
            shutil.copy2(meta_src, tmp_out / "metadata.yaml")
        _validate_openvino_dir(tmp_out)
        _move_artifact(tmp_out, output_dir)

    return _validate_openvino_dir(output_dir)


def export_stage_models(
    weights_path: Path,
    quantization_dir: Path,
    calibration_images: list[Path],
    calibration_yaml: Path,
    *,
    imgsz: int,
    model_stem: str = YOLO_MODEL_STEM,
    calibrate_method: str = "MinMax",
) -> dict[str, Path]:
    """Exports all CPU ablation stages for a fixed square input resolution."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"PyTorch weights not found: {weights_path}")

    paths = get_quantized_model_paths(Path(quantization_dir), imgsz, model_stem=model_stem)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    if weights_path.resolve() != paths["pt"].resolve():
        shutil.copy2(weights_path, paths["pt"])

    onnx_fp32 = export_onnx_fp32(paths["pt"], paths["onnx_fp32"], imgsz=imgsz)
    onnx_int8 = quantize_onnx_int8(
        onnx_fp32,
        paths["onnx_int8"],
        calibration_images,
        imgsz=imgsz,
        calibrate_method=calibrate_method,
    )
    ov_fp32 = export_openvino_fp32(paths["pt"], paths["openvino_fp32"], imgsz=imgsz)
    ov_int8 = export_openvino_int8(
        paths["pt"],
        paths["openvino_int8"],
        calibration_yaml,
        imgsz=imgsz,
    )

    return {
        "pytorch_fp32": paths["pt"],
        "onnx_fp32": onnx_fp32,
        "onnx_int8": onnx_int8,
        "openvino_fp32": ov_fp32,
        "openvino_int8": ov_int8,
    }


def _is_valid_stage_artifact(stage_name: str, artifact_path: Path) -> bool:
    if stage_name.startswith("openvino_"):
        try:
            _validate_openvino_dir(artifact_path)
            return True
        except FileNotFoundError:
            return False
    return artifact_path.is_file() and artifact_path.stat().st_size > 0


def _export_single_stage(
    stage_name: str,
    weights_path: Path,
    quantization_dir: Path,
    calibration_images: list[Path],
    calibration_yaml: Path,
    *,
    imgsz: int,
    model_stem: str = YOLO_MODEL_STEM,
    calibrate_method: str = "MinMax",
    skip_existing: bool = False,
) -> tuple[Path, dict[str, object]]:
    paths = get_quantized_model_paths(Path(quantization_dir), imgsz, model_stem=model_stem)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    meta_info: dict[str, object] = {}

    if stage_name == "pytorch_fp32":
        if not skip_existing or not _is_valid_stage_artifact(stage_name, paths["pt"]):
            if weights_path.resolve() != paths["pt"].resolve():
                shutil.copy2(weights_path, paths["pt"])
        return paths["pt"], meta_info

    if stage_name == "onnx_fp32":
        if skip_existing and _is_valid_stage_artifact("onnx_fp32", paths["onnx_fp32"]):
            return paths["onnx_fp32"], meta_info
        return export_onnx_fp32(weights_path, paths["onnx_fp32"], imgsz=imgsz), meta_info

    if stage_name == "onnx_int8":
        if not _is_valid_stage_artifact("onnx_fp32", paths["onnx_fp32"]) or not skip_existing:
            if not _is_valid_stage_artifact("onnx_fp32", paths["onnx_fp32"]):
                export_onnx_fp32(weights_path, paths["onnx_fp32"], imgsz=imgsz)
        if skip_existing and _is_valid_stage_artifact("onnx_int8", paths["onnx_int8"]):
            excluded = select_onnx_nodes_to_exclude(onnx.load(str(paths["onnx_fp32"])))
            meta_info["excluded_nodes"] = excluded
            meta_info["onnx_fallback_full_head"] = False
            return paths["onnx_int8"], meta_info
        out_path, excluded, used_fallback = _quantize_onnx_with_fallback(
            paths["onnx_fp32"],
            paths["onnx_int8"],
            calibration_images,
            imgsz=imgsz,
            calibrate_method=calibrate_method,
        )
        meta_info["excluded_nodes"] = excluded
        meta_info["onnx_fallback_full_head"] = used_fallback
        return out_path, meta_info

    if stage_name == "openvino_fp32":
        if skip_existing and _is_valid_stage_artifact("openvino_fp32", paths["openvino_fp32"]):
            return paths["openvino_fp32"], meta_info
        return export_openvino_fp32(weights_path, paths["openvino_fp32"], imgsz=imgsz), meta_info

    if stage_name == "openvino_int8":
        if skip_existing and _is_valid_stage_artifact("openvino_int8", paths["openvino_int8"]):
            return paths["openvino_int8"], meta_info
        return (
            export_openvino_int8(
                weights_path,
                paths["openvino_int8"],
                calibration_yaml,
                imgsz=imgsz,
            ),
            meta_info,
        )

    raise ValueError(f"Unknown quantization stage: {stage_name!r}")


_INT8_TO_FP32_BASELINE = {
    "onnx_int8": "onnx_fp32",
    "openvino_int8": "openvino_fp32",
}

_ALLOWED_REPORT_SPLITS = frozenset({
    "val",
    "test",
    "ptq_ablation_val",
    "ptq_ablation_test",
})


def _resolve_fp32_baseline_stage(stage: str) -> str | None:
    if stage in _INT8_TO_FP32_BASELINE:
        return _INT8_TO_FP32_BASELINE[stage]
    if stage.startswith("onnx_int8"):
        return "onnx_fp32"
    if stage.startswith("openvino_int8"):
        return "openvino_fp32"
    return None


def _load_yolo_for_artifact(artifact_path: Path) -> YOLO:
    if artifact_path.suffix.lower() == ".pt":
        return YOLO(str(artifact_path))
    return YOLO(str(artifact_path), task="detect")


def _get_process_memory_mb() -> tuple[float, float]:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _MemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            get_mem_info = getattr(kernel32, "K32GetProcessMemoryInfo", None)
            if get_mem_info is None:
                get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_mem_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_MemoryCounters),
                wintypes.DWORD,
            ]
            get_mem_info.restype = wintypes.BOOL

            counters = _MemoryCounters()
            counters.cb = ctypes.sizeof(_MemoryCounters)
            handle = kernel32.GetCurrentProcess()
            if get_mem_info(handle, ctypes.byref(counters), counters.cb):
                mb = 1024 * 1024
                return counters.WorkingSetSize / mb, counters.PeakWorkingSetSize / mb
        except Exception:
            return 0.0, 0.0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        mb_val = float(usage) / divisor
        return mb_val, mb_val
    except Exception:
        return 0.0, 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_runtime_versions() -> dict[str, str]:
    packages = (
        "torch",
        "ultralytics",
        "onnx",
        "onnxslim",
        "onnxruntime",
        "openvino",
        "nncf",
        "opencv-python",
        "numpy",
    )
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def _to_json_safe(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    return value


def _bgr_to_nchw(bgr: np.ndarray, imgsz: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded, _ = letterbox_image(rgb, imgsz=imgsz)
    chw = np.transpose(padded, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None, ...])


def _load_benchmark_frames(
    dataset_root: Path,
    split: str,
    imgsz: int,
    max_frames: int,
    *,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if split not in {"val", "test"}:
        raise ValueError(f"Benchmark split must be 'val' or 'test', got {split!r}")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    active = select_manifest_frames(
        Path(dataset_root),
        split=split,
        manifest_path=manifest_path,
        source_path=source_path,
        allowed_splits=("val", "test"),
    )
    sample_count = min(max_frames, len(active))
    indices = np.linspace(0, len(active) - 1, sample_count, dtype=int)

    bgr_frames = []
    nchw_tensors = []
    for idx in indices:
        img_path, _ = active[int(idx)]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Unreadable benchmark image: {img_path}")
        bgr_frames.append(bgr)
        nchw_tensors.append(_bgr_to_nchw(bgr, imgsz=imgsz))

    return bgr_frames, nchw_tensors


def _summarize_latencies_ms(durations_s: list[float]) -> dict[str, float]:
    ms = np.asarray(durations_s, dtype=float) * 1000.0
    mean_ms = float(np.mean(ms))
    p50_ms = float(np.percentile(ms, 50))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(ms)),
        "p50_ms": p50_ms,
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "fps_p50": 1000.0 / p50_ms if p50_ms > 0 else 0.0,
    }


def _time_inference_loop(runner, samples: list, *, n_warmup: int, n_benchmark: int) -> dict[str, float]:
    for idx in range(n_warmup):
        runner(samples[idx % len(samples)])

    durations_s = []
    for idx in range(n_benchmark):
        sample = samples[idx % len(samples)]
        t0 = time.perf_counter()
        runner(sample)
        durations_s.append(time.perf_counter() - t0)

    return _summarize_latencies_ms(durations_s)


def _build_onnx_compute_runner(artifact_path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    def _run(tensor: np.ndarray) -> None:
        session.run(output_names, {input_name: tensor})

    return _run


def _build_openvino_compute_runner(artifact_path: Path):
    import openvino as ov

    _validate_openvino_dir(artifact_path)
    xml_path = next(sorted(artifact_path.glob("*.xml")).__iter__())
    core = ov.Core()
    ov_model = core.read_model(model=str(xml_path), weights=str(xml_path.with_suffix(".bin")))
    compiled = core.compile_model(
        ov_model,
        device_name="CPU",
        config={"PERFORMANCE_HINT": "LATENCY"},
    )

    def _run(tensor: np.ndarray) -> None:
        compiled(tensor)

    return _run


def _build_pytorch_compute_runner(artifact_path: Path):
    yolo_model = _load_yolo_for_artifact(artifact_path)
    module = yolo_model.model
    if hasattr(module, "float"):
        module = module.float()
    if hasattr(module, "to"):
        module = module.to("cpu")
    if hasattr(module, "eval"):
        module = module.eval()
    if hasattr(module, "fuse") and callable(module.fuse):
        try:
            module = module.fuse()
        except Exception:
            pass

    def _run(tensor: np.ndarray) -> None:
        torch_input = torch.from_numpy(tensor)
        with torch.inference_mode():
            module(torch_input)

    return _run


def _build_compute_runner(artifact_path: Path):
    if artifact_path.is_dir():
        return _build_openvino_compute_runner(artifact_path)
    suffix = artifact_path.suffix.lower()
    if suffix == ".onnx":
        return _build_onnx_compute_runner(artifact_path)
    if suffix == ".pt":
        return _build_pytorch_compute_runner(artifact_path)
    raise ValueError(f"Unsupported artifact for CPU benchmark: {artifact_path}")


def benchmark_stage_model(
    artifact_path: Path,
    bgr_frames: list[np.ndarray],
    nchw_tensors: list[np.ndarray],
    *,
    imgsz: int,
    n_warmup: int = 10,
    n_benchmark: int = 100,
    e2e_conf: float = 0.25,
) -> dict[str, float]:
    """Benchmarks compute-only and end-to-end CPU latency plus process RSS memory."""
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Benchmark artifact not found: {artifact_path}")
    if not bgr_frames or len(bgr_frames) != len(nchw_tensors):
        raise ValueError("bgr_frames and nchw_tensors must be non-empty and equal length")
    if n_warmup < 0 or n_benchmark <= 0:
        raise ValueError(f"Invalid benchmark counts: warmup={n_warmup}, benchmark={n_benchmark}")

    compute_runner = _build_compute_runner(artifact_path)
    compute_stats = _time_inference_loop(
        compute_runner,
        nchw_tensors,
        n_warmup=n_warmup,
        n_benchmark=n_benchmark,
    )

    yolo_model = _load_yolo_for_artifact(artifact_path)

    def _run_e2e(frame: np.ndarray) -> None:
        yolo_model.predict(
            frame,
            imgsz=imgsz,
            conf=e2e_conf,
            device="cpu",
            rect=False,
            verbose=False,
        )

    e2e_stats = _time_inference_loop(
        _run_e2e,
        bgr_frames,
        n_warmup=n_warmup,
        n_benchmark=n_benchmark,
    )
    rss_mb, peak_rss_mb = _get_process_memory_mb()

    return {
        **{f"compute_{k}": v for k, v in compute_stats.items()},
        **{f"e2e_{k}": v for k, v in e2e_stats.items()},
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
    }


def _compute_raw_detection_diagnostics(
    raw_database: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    tp_confs: list[float] = []
    total_gt = 0
    for entry in raw_database:
        gt_boxes = entry["gt_boxes"]
        total_gt += len(gt_boxes)
        if not entry["is_positive"] or not gt_boxes:
            continue
        tp_indices, _, _ = evaluate_predictions(
            gt_boxes,
            entry["pred_boxes"],
            iou_threshold=iou_threshold,
        )
        tp_confs.extend(float(entry["pred_boxes"][idx][4]) for idx in tp_indices)

    if total_gt == 0:
        nan = float("nan")
        return {
            "total_gt": 0,
            "raw_recall": nan,
            "tp_conf_p10": nan,
            "tp_conf_median": nan,
            "tp_conf_p90": nan,
            "tp_recall_ge_0_30": nan,
        }

    if not tp_confs:
        return {
            "total_gt": total_gt,
            "raw_recall": 0.0,
            "tp_conf_p10": float("nan"),
            "tp_conf_median": float("nan"),
            "tp_conf_p90": float("nan"),
            "tp_recall_ge_0_30": 0.0,
        }

    arr = np.asarray(tp_confs, dtype=float)
    return {
        "total_gt": total_gt,
        "raw_recall": float(len(arr) / total_gt),
        "tp_conf_p10": float(np.percentile(arr, 10)),
        "tp_conf_median": float(np.median(arr)),
        "tp_conf_p90": float(np.percentile(arr, 90)),
        "tp_recall_ge_0_30": float(np.sum(arr >= 0.30) / total_gt),
    }


def _filter_raw_database(raw_database: list[dict], min_conf: float) -> list[dict]:
    return [
        {
            **entry,
            "pred_boxes": [box for box in entry["pred_boxes"] if float(box[4]) >= min_conf],
        }
        for entry in raw_database
    ]


def _compute_coco_map_metrics(raw_database: list[dict]) -> tuple[float, float]:
    aps = []
    for iou in np.round(np.linspace(0.50, 0.95, 10), 2):
        gt_scale_counts, scores_arr, tps_arr, _, _ = _collect_matched_detections(
            raw_database,
            float(iou),
        )
        total_gt = sum(gt_scale_counts.values())
        ap = _compute_average_precision(scores_arr, tps_arr, total_gt)
        if math.isnan(ap):
            return float("nan"), float("nan")
        aps.append(ap)
    return float(aps[0]), float(np.mean(aps))


def evaluate_quantized_stage(
    artifact_path: Path,
    dataset_root: Path,
    eval_yaml: Path | None = None,
    stage_name: str = "model",
    *,
    imgsz: int,
    eval_split: str = "val",
    collection_conf: float = 0.01,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
) -> dict[str, object]:
    """Runs single-pass threshold-free CPU evaluation (mAP50/50-95 + scale/negative metrics)."""
    if eval_split not in {"val", "test"}:
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")

    yolo_model = _load_yolo_for_artifact(Path(artifact_path))
    raw_conf = min(float(collection_conf), 0.001)
    raw_db = run_dataset_predictions_db(
        yolo_model,
        Path(dataset_root),
        imgsz=imgsz,
        collection_conf=raw_conf,
        split=eval_split,
        manifest_path=manifest_path,
        source_path=source_path,
        allowed_splits=("val", "test"),
        device="cpu",
    )
    map50, map50_95 = _compute_coco_map_metrics(raw_db)
    slice_db = _filter_raw_database(raw_db, collection_conf) if raw_conf < collection_conf else raw_db
    tf_report = evaluate_threshold_free_slices(
        slice_db,
        iou_threshold=0.5,
        target_recall=0.85,
        max_neg_fpr=5.0,
    )
    summary = tf_report["summary"].iloc[0]
    by_scale_df = tf_report["by_scale"]
    by_scale_idx = by_scale_df.set_index("Scale")
    tp_diag = _compute_raw_detection_diagnostics(slice_db)

    metrics = {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "AP50": float(summary["AP50"]),
        "AP50_small": float(summary["AP50_small"]),
        "AP50_medium": float(summary["AP50_medium"]),
        "AP50_large": float(summary["AP50_large"]),
        "Recall_small @ Neg FPR<=5%": float(summary["Recall_small @ Neg FPR<=5%"]),
        "Neg FPR @ R>=0.85 (%)": float(summary["Neg FPR @ R>=0.85 (%)"]),
        "Mean Neg FA Rate (%)": float(summary["Mean Neg FA Rate (%)"]),
        "Max-F1 P": float(summary["Max-F1 P"]),
        "Max-F1 R": float(summary["Max-F1 R"]),
        "Max-F1 Recall_small": float(summary["Max-F1 Recall_small"]),
        "Max-F1 Neg FA Rate (%)": float(summary["Max-F1 Neg FA Rate (%)"]),
        "Recall_small (raw)": float(by_scale_idx.loc["small", "Max Recall (raw)"])
        if "small" in by_scale_idx.index
        else float("nan"),
        "Max Recall (raw)": tp_diag["raw_recall"],
        "tp_conf_p10": tp_diag["tp_conf_p10"],
        "tp_conf_median": tp_diag["tp_conf_median"],
        "tp_conf_p90": tp_diag["tp_conf_p90"],
        "tp_recall_ge_0_30": tp_diag["tp_recall_ge_0_30"],
    }
    curves = {
        "thresholds": tf_report["thresholds"],
        "precision": tf_report["precision"],
        "recall": tf_report["recall"],
        "recall_small": tf_report["recall_small"],
        "f1": tf_report["f1"],
        "neg_fa_rate": tf_report["neg_fa_rate"],
    }
    return {
        "metrics": metrics,
        "by_scale": by_scale_df.to_dict(orient="records"),
        "curves": curves,
    }


def _empty_eval_metrics() -> dict[str, float]:
    nan = float("nan")
    return {
        "mAP50": nan,
        "mAP50-95": nan,
        "AP50": nan,
        "AP50_small": nan,
        "AP50_medium": nan,
        "AP50_large": nan,
        "Recall_small @ Neg FPR<=5%": nan,
        "Neg FPR @ R>=0.85 (%)": nan,
        "Mean Neg FA Rate (%)": nan,
        "Max-F1 P": nan,
        "Max-F1 R": nan,
        "Max-F1 Recall_small": nan,
        "Max-F1 Neg FA Rate (%)": nan,
        "Recall_small (raw)": nan,
        "Max Recall (raw)": nan,
        "tp_conf_p10": nan,
        "tp_conf_median": nan,
        "tp_conf_p90": nan,
        "tp_recall_ge_0_30": nan,
    }


def _empty_benchmark_metrics() -> dict[str, float]:
    nan = float("nan")
    return {
        "compute_mean_ms": nan,
        "compute_std_ms": nan,
        "compute_p50_ms": nan,
        "compute_p95_ms": nan,
        "compute_p99_ms": nan,
        "compute_fps": nan,
        "compute_fps_p50": nan,
        "e2e_mean_ms": nan,
        "e2e_std_ms": nan,
        "e2e_p50_ms": nan,
        "e2e_p95_ms": nan,
        "e2e_p99_ms": nan,
        "e2e_fps": nan,
        "e2e_fps_p50": nan,
        "rss_mb": nan,
        "peak_rss_mb": nan,
    }


def _safe_diff(a: float, b: float) -> float:
    return float(a - b) if not (math.isnan(a) or math.isnan(b)) else float("nan")


def _safe_ratio(num: float, den: float) -> float:
    if math.isnan(num) or math.isnan(den) or den <= 0:
        return float("nan")
    return float(num / den)


def _annotate_ablation_deltas(rows: list[dict[str, object]]) -> pd.DataFrame:
    by_key = {(int(r["imgsz"]), str(r["stage"])): r for r in rows}
    annotated = []

    for row in rows:
        item = dict(row)
        imgsz = int(row["imgsz"])
        stage = str(row["stage"])
        pt_raw = by_key.get((imgsz, "pytorch_fp32"))
        pt_ref = pt_raw if pt_raw is not None and pt_raw.get("status", "ok") == "ok" else None
        fp32_name = _resolve_fp32_baseline_stage(stage)
        fp32_raw = by_key.get((imgsz, fp32_name)) if fp32_name else None
        fp32_ref = fp32_raw if fp32_raw is not None and fp32_raw.get("status", "ok") == "ok" else None

        item["speedup_compute_vs_pt"] = (
            _safe_ratio(float(pt_ref["compute_mean_ms"]), float(item["compute_mean_ms"]))
            if pt_ref is not None
            else float("nan")
        )
        item["speedup_e2e_vs_pt"] = (
            _safe_ratio(float(pt_ref["e2e_mean_ms"]), float(item["e2e_mean_ms"]))
            if pt_ref is not None
            else float("nan")
        )
        item["size_ratio_vs_pt"] = (
            _safe_ratio(float(item["size_mb"]), float(pt_ref["size_mb"]))
            if pt_ref is not None
            else float("nan")
        )

        if fp32_ref is not None:
            item["ap50_delta_vs_fp32"] = _safe_diff(float(item["AP50"]), float(fp32_ref["AP50"]))
            item["ap50_small_delta_vs_fp32"] = _safe_diff(
                float(item["AP50_small"]),
                float(fp32_ref["AP50_small"]),
            )
            item["map50_delta_vs_fp32"] = _safe_diff(float(item["mAP50"]), float(fp32_ref["mAP50"]))
            item["map50_95_delta_vs_fp32"] = _safe_diff(
                float(item["mAP50-95"]),
                float(fp32_ref["mAP50-95"]),
            )
            item["neg_fpr_at_r85_delta_vs_fp32"] = _safe_diff(
                float(item["Neg FPR @ R>=0.85 (%)"]),
                float(fp32_ref["Neg FPR @ R>=0.85 (%)"]),
            )
            item["speedup_compute_vs_fp32"] = _safe_ratio(
                float(fp32_ref["compute_mean_ms"]),
                float(item["compute_mean_ms"]),
            )
            item["speedup_e2e_vs_fp32"] = _safe_ratio(
                float(fp32_ref["e2e_mean_ms"]),
                float(item["e2e_mean_ms"]),
            )
            ratio = _safe_ratio(float(item["size_mb"]), float(fp32_ref["size_mb"]))
            item["size_reduction_pct_vs_fp32"] = (
                (1.0 - ratio) * 100.0 if not math.isnan(ratio) else float("nan")
            )
        else:
            for col in (
                "ap50_delta_vs_fp32",
                "ap50_small_delta_vs_fp32",
                "map50_delta_vs_fp32",
                "map50_95_delta_vs_fp32",
                "neg_fpr_at_r85_delta_vs_fp32",
                "speedup_compute_vs_fp32",
                "speedup_e2e_vs_fp32",
                "size_reduction_pct_vs_fp32",
            ):
                item[col] = float("nan")

        annotated.append(item)

    return pd.DataFrame(annotated)


def save_quantization_reports(
    results_df: pd.DataFrame,
    quantization_dir: Path,
    metadata: dict[str, object],
    *,
    eval_split: str,
) -> tuple[Path, Path]:
    """Saves CPU ablation CSV and full JSON metadata under reports/<eval_split>."""
    if eval_split not in _ALLOWED_REPORT_SPLITS:
        raise ValueError(f"eval_split must be one of {sorted(_ALLOWED_REPORT_SPLITS)}, got {eval_split!r}")

    report_dir = Path(quantization_dir) / "reports" / eval_split
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "ablation_results.csv"
    metadata_path = report_dir / "quantization_metadata.json"

    results_df.to_csv(csv_path, index=False)
    safe_metadata = _to_json_safe(metadata)
    metadata_path.write_text(
        json.dumps(safe_metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return csv_path, metadata_path


def _disable_hf_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    hf_constants.HF_HUB_DISABLE_XET = True


def _prepare_hf_upload_client(
    repo_id: str | None = None,
    token: str | None = None,
) -> tuple[str, HfApi]:
    _disable_hf_xet()
    target_repo, target_token = verify_hf_access(repo_id=repo_id, token=token)
    return target_repo, HfApi(token=target_token)


def _run_with_timeout(fn, timeout_seconds: float | None, *args, **kwargs):
    if timeout_seconds is None or timeout_seconds <= 0:
        return fn(*args, **kwargs)

    outcome: queue.Queue = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            outcome.put((True, fn(*args, **kwargs)))
        except BaseException as exc:
            outcome.put((False, exc))

    worker = threading.Thread(target=_worker, name="hf-upload-worker", daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(f"Timed out after {timeout_seconds:.0f}s")

    ok, payload = outcome.get_nowait()
    if not ok:
        assert isinstance(payload, BaseException)
        raise payload
    return payload


def upload_quantization_artifacts_to_hf(
    quantization_dir: Path,
    *,
    eval_split: str,
    resolutions: tuple[int, ...] = QUANTIZATION_RESOLUTIONS,
    model_stem: str = YOLO_MODEL_STEM,
    tag: str | None = None,
    repo_id: str | None = None,
    token: str | None = None,
) -> None:
    """Uploads only quantized INT8 models (ONNX + OpenVINO) and split reports to HF."""
    target_repo, api = _prepare_hf_upload_client(repo_id=repo_id, token=token)
    quantization_dir = Path(quantization_dir)

    allow_patterns = [f"reports/{eval_split}/**"]
    required_files = [
        quantization_dir / "reports" / eval_split / "ablation_results.csv",
        quantization_dir / "reports" / eval_split / "quantization_metadata.json",
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Report file missing for HF upload: {path}")

    for imgsz in resolutions:
        paths = get_quantized_model_paths(quantization_dir, imgsz, model_stem=model_stem)
        if not paths["onnx_int8"].is_file():
            raise FileNotFoundError(f"ONNX INT8 artifact missing for HF upload: {paths['onnx_int8']}")
        _validate_openvino_dir(paths["openvino_int8"])
        allow_patterns.append(f"{imgsz}/{paths['onnx_int8'].name}")
        allow_patterns.append(f"{imgsz}/{paths['openvino_int8'].name}/**")
    api.upload_folder(
        folder_path=str(quantization_dir),
        path_in_repo="models/quantization",
        repo_id=target_repo,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        commit_message=f"Upload CPU INT8 models and {eval_split} quantization reports",
    )
    if tag is not None and tag.strip():
        api.create_tag(
            repo_id=target_repo,
            tag=tag.strip(),
            repo_type="dataset",
        )


def upload_single_quantized_artifact_to_hf(
    artifact_path: Path,
    imgsz: int,
    stage_name: str,
    *,
    repo_id: str | None = None,
    token: str | None = None,
) -> None:
    """Immediately uploads a single quantized INT8 model (file or OpenVINO dir) to HF."""
    target_repo, api = _prepare_hf_upload_client(repo_id=repo_id, token=token)
    artifact_path = Path(artifact_path)
    path_in_repo = f"models/quantization/{imgsz}/{artifact_path.name}"
    commit_msg = f"Upload {stage_name} ({imgsz}) quantized model"

    if artifact_path.is_dir():
        _validate_openvino_dir(artifact_path)
        api.upload_folder(
            folder_path=str(artifact_path),
            path_in_repo=path_in_repo,
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        return

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Quantized artifact not found for HF upload: {artifact_path}")
    api.upload_file(
        path_or_fileobj=str(artifact_path),
        path_in_repo=path_in_repo,
        repo_id=target_repo,
        repo_type="dataset",
        commit_message=commit_msg,
    )


def upload_quantization_reports_to_hf(
    quantization_dir: Path,
    *,
    eval_split: str,
    tag: str | None = None,
    repo_id: str | None = None,
    token: str | None = None,
) -> None:
    """Uploads split CSV, JSON metadata, and logs under models/quantization/reports/<split>."""
    target_repo, api = _prepare_hf_upload_client(repo_id=repo_id, token=token)
    report_dir = Path(quantization_dir) / "reports" / eval_split
    for name in ("ablation_results.csv", "quantization_metadata.json"):
        if not (report_dir / name).is_file():
            raise FileNotFoundError(f"Report file missing for HF upload: {report_dir / name}")

    api.upload_folder(
        folder_path=str(report_dir),
        path_in_repo=f"models/quantization/reports/{eval_split}",
        repo_id=target_repo,
        repo_type="dataset",
        commit_message=f"Upload {eval_split} quantization reports",
    )
    if tag is not None and tag.strip():
        api.create_tag(
            repo_id=target_repo,
            tag=tag.strip(),
            repo_type="dataset",
        )


def _try_hf_operation(
    action_name: str,
    fn,
    *args,
    timeout_seconds: float | None = HF_UPLOAD_TIMEOUT_SECONDS,
    **kwargs,
) -> bool:
    try:
        _disable_hf_xet()
        _run_with_timeout(fn, timeout_seconds, *args, **kwargs)
        return True
    except (ValueError, PermissionError):
        raise
    except Exception as exc:
        log.warning("HF %s failed (%s); continuing locally.", action_name, exc)
        return False


def _clean_checkpoint_row(row: dict[str, object]) -> dict[str, object]:
    return {
        k: (
            float("nan")
            if v is None and k not in {"artifact", "error", "stage", "status", "hf_uploaded"}
            else v
        )
        for k, v in row.items()
    }


def _load_resume_checkpoint(
    quantization_dir: Path,
    eval_split: str,
    *,
    source_sha256: str,
    max_calibration_images: int,
    seed: int,
    calibrate_method: str,
) -> tuple[
    dict[tuple[int, str], dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, list[str]],
    dict[str, bool],
]:
    meta_path = Path(quantization_dir) / "reports" / eval_split / "quantization_metadata.json"
    if not meta_path.is_file():
        return {}, {}, {}, {}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse checkpoint metadata %s (%s); starting fresh.", meta_path, exc)
        return {}, {}, {}, {}

    cal = meta.get("calibration") or {}
    onnx_cfg = meta.get("onnx_quantization") or {}
    if (
        meta.get("source_weights_sha256") != source_sha256
        or cal.get("max_images") != max_calibration_images
        or cal.get("seed") != seed
        or onnx_cfg.get("calibrate_method") != calibrate_method
    ):
        log.warning("Resume checkpoint config mismatch in %s; starting fresh.", meta_path)
        return {}, {}, {}, {}


    rows_by_key: dict[tuple[int, str], dict[str, object]] = {}
    for row in meta.get("summary_rows") or []:
        if row.get("status") != "ok":
            continue
        artifact = Path(str(row.get("artifact", "")))
        stage_name = str(row["stage"])
        if not _is_valid_stage_artifact(stage_name, artifact):
            continue
        rows_by_key[(int(row["imgsz"]), stage_name)] = _clean_checkpoint_row(row)

    details = dict(meta.get("stage_details") or {})
    excluded = dict(onnx_cfg.get("excluded_nodes_by_resolution") or {})
    fallbacks = dict(onnx_cfg.get("fallback_full_head_by_resolution") or {})
    return rows_by_key, details, excluded, fallbacks


def _ordered_rows_snapshot(
    rows_by_key: dict[tuple[int, str], dict[str, object]],
    stage_order: tuple[str, ...] = ALL_STAGES,
) -> list[dict[str, object]]:
    order_map = {name: idx for idx, name in enumerate(stage_order)}

    def _sort_key(key: tuple[int, str]) -> tuple[int, int, str]:
        imgsz, stage_name = key
        return -imgsz, order_map.get(stage_name, len(order_map)), stage_name

    return [rows_by_key[key] for key in sorted(rows_by_key, key=_sort_key)]


def _measure_stage_artifact(
    artifact_path: Path,
    dataset_root: Path,
    *,
    stage_name: str,
    imgsz: int,
    bgr_frames: list[np.ndarray],
    nchw_tensors: list[np.ndarray],
    eval_split: str,
    n_warmup: int,
    n_benchmark: int,
    collection_conf: float,
    manifest_path: Path,
    source_path: Path | None,
    run_eval: bool = True,
) -> tuple[dict[str, object], dict[str, object] | None]:
    size_mb = get_artifact_size_mb(artifact_path)
    bench = benchmark_stage_model(
        artifact_path,
        bgr_frames,
        nchw_tensors,
        imgsz=imgsz,
        n_warmup=n_warmup,
        n_benchmark=n_benchmark,
    )

    detail: dict[str, object] | None = None
    if run_eval:
        eval_data = evaluate_quantized_stage(
            artifact_path,
            Path(dataset_root),
            None,
            stage_name,
            imgsz=imgsz,
            eval_split=eval_split,
            collection_conf=collection_conf,
            manifest_path=manifest_path,
            source_path=source_path,
        )
        eval_metrics = eval_data["metrics"]
        detail = {
            "by_scale": eval_data["by_scale"],
            "curves": eval_data["curves"],
        }
    else:
        eval_metrics = _empty_eval_metrics()

    row: dict[str, object] = {
        "imgsz": imgsz,
        "stage": stage_name,
        "status": "ok",
        "hf_uploaded": False,
        "error": "",
        "artifact": str(artifact_path),
        "size_mb": size_mb,
        **bench,
        **eval_metrics,
    }
    log.info(
        "[STAGE OK] %d %s | e2e=%.2fms (%.1ffps) | mAP50=%.4f | AP50=%.4f | AP50_small=%.4f",
        imgsz,
        stage_name,
        bench["e2e_mean_ms"],
        bench["e2e_fps"],
        eval_metrics["mAP50"] if not math.isnan(eval_metrics["mAP50"]) else 0.0,
        eval_metrics["AP50"] if not math.isnan(eval_metrics["AP50"]) else 0.0,
        eval_metrics["AP50_small"] if not math.isnan(eval_metrics["AP50_small"]) else 0.0,
    )
    return row, detail


def _failed_stage_row(imgsz: int, stage_name: str, error: str) -> dict[str, object]:
    return {
        "imgsz": imgsz,
        "stage": stage_name,
        "status": "failed",
        "hf_uploaded": False,
        "error": error,
        "artifact": "",
        "size_mb": float("nan"),
        **_empty_benchmark_metrics(),
        **_empty_eval_metrics(),
    }


def run_cpu_quantization_ablation(
    dataset_root: Path,
    *,
    weights_path: Path | None = None,
    quantization_dir: Path | None = None,
    working_dir: Path | None = None,
    resolutions: tuple[int, ...] = QUANTIZATION_RESOLUTIONS,
    stages: tuple[str, ...] = ALL_STAGES,
    eval_split: str = "val",
    eval_pytorch: bool = False,
    resume: bool = False,
    skip_existing_exports: bool = False,
    max_calibration_images: int = DEFAULT_CALIBRATION_IMAGES,
    calibrate_method: str = "MinMax",
    seed: int = 42,
    n_warmup: int = 10,
    n_benchmark: int = 100,
    collection_conf: float = 0.01,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    model_stem: str = YOLO_MODEL_STEM,
    upload_to_hf: bool = False,
    hf_tag: str | None = None,
    repo_id: str | None = None,
    token: str | None = None,
) -> dict[str, object]:
    """Runs resumable single-pass CPU quantization, benchmark, threshold-free eval, and checkpoints."""
    if eval_split not in {"val", "test"}:
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")
    if not resolutions or any(r <= 0 for r in resolutions):
        raise ValueError(f"resolutions must be non-empty positive ints, got {resolutions!r}")
    if not stages or any(s not in ALL_STAGES for s in stages):
        raise ValueError(f"Invalid stages {stages!r}; expected subset of {ALL_STAGES}")

    if upload_to_hf:
        _try_hf_operation(
            "preflight check",
            verify_hf_access,
            repo_id=repo_id,
            token=token,
            timeout_seconds=min(30.0, HF_UPLOAD_TIMEOUT_SECONDS),
        )

    target_quant_dir = Path(quantization_dir) if quantization_dir is not None else config.QUANTIZATION_DIR
    target_working_dir = Path(working_dir) if working_dir is not None else config.WORKING_DIR
    cache_dir = target_working_dir / "hf_cache" if working_dir is not None else config.LOCAL_HF_DATA_DIR
    source_pt = resolve_source_weights(
        weights_path,
        cache_dir=cache_dir if weights_path is None else None,
        repo_id=repo_id,
        token=token,
    )
    source_sha256 = _file_sha256(source_pt)

    rows_by_key: dict[tuple[int, str], dict[str, object]] = {}
    per_stage_details: dict[str, dict[str, object]] = {}
    onnx_excluded_nodes: dict[str, list[str]] = {}
    onnx_fallback_flags: dict[str, bool] = {}

    if resume:
        rows_by_key, per_stage_details, onnx_excluded_nodes, onnx_fallback_flags = _load_resume_checkpoint(
            target_quant_dir,
            eval_split,
            source_sha256=source_sha256,
            max_calibration_images=max_calibration_images,
            seed=seed,
            calibrate_method=calibrate_method,
        )

    effective_skip_exports = resume or skip_existing_exports
    cal_run_name = "calibration_ptq"
    calibration_yaml, calibration_images = prepare_calibration_dataset(
        Path(dataset_root),
        target_working_dir,
        manifest_path=manifest_path,
        source_path=source_path,
        max_images=max_calibration_images,
        seed=seed,
        run_name=cal_run_name,
        overwrite=True,
    )

    artifacts_by_res: dict[int, dict[str, Path]] = {}

    def _build_metadata(results_df: pd.DataFrame, *, complete: bool, failed_items: list[dict[str, object]]) -> dict[str, object]:
        return {
            "source_weights": str(source_pt),
            "source_weights_sha256": source_sha256,
            "eval_split": eval_split,
            "eval_pytorch": eval_pytorch,
            "resolutions": list(resolutions),
            "stages": list(stages),
            "collection_conf": collection_conf,
            "complete": complete,
            "failed_stages": failed_items,
            "benchmark": {
                "n_warmup": n_warmup,
                "n_benchmark": n_benchmark,
                "split": eval_split,
            },
            "calibration": {
                "manifest_path": str(manifest_path),
                "max_images": max_calibration_images,
                "actual_images": len(calibration_images),
                "seed": seed,
                "filenames": [path.name for path in calibration_images],
            },
            "onnx_quantization": {
                "format": "QDQ",
                "activation_type": "QUInt8",
                "weight_type": "QInt8",
                "per_channel": True,
                "reduce_range": False,
                "calibrate_method": calibrate_method,
                "excluded_nodes_by_resolution": onnx_excluded_nodes,
                "fallback_full_head_by_resolution": onnx_fallback_flags,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "packages": _extract_runtime_versions(),
            },
            "stage_details": per_stage_details,
            "summary_rows": results_df.to_dict(orient="records"),
        }

    def _save_checkpoint(
        *,
        complete: bool | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, object]], Path, Path]:
        snapshot = _ordered_rows_snapshot(rows_by_key)
        snapshot_df = _annotate_ablation_deltas(snapshot)
        failed_items = [
            {"imgsz": int(r["imgsz"]), "stage": str(r["stage"]), "error": str(r.get("error", ""))}
            for r in snapshot
            if r.get("status") != "ok"
        ]
        is_complete = (len(failed_items) == 0) if complete is None else complete
        saved_csv, saved_meta = save_quantization_reports(
            snapshot_df,
            target_quant_dir,
            _build_metadata(snapshot_df, complete=is_complete, failed_items=failed_items),
            eval_split=eval_split,
        )
        return snapshot_df, failed_items, saved_csv, saved_meta

    try:
        for imgsz in resolutions:
            artifacts_by_res.setdefault(imgsz, {})
            bgr_frames = None
            nchw_tensors = None

            for stage_name in stages:
                is_int8_stage = stage_name in {"onnx_int8", "openvino_int8"}
                if (imgsz, stage_name) in rows_by_key:
                    existing_row = rows_by_key[(imgsz, stage_name)]
                    existing_artifact = Path(str(existing_row["artifact"]))
                    artifacts_by_res[imgsz][stage_name] = existing_artifact
                    if upload_to_hf and is_int8_stage and not bool(existing_row.get("hf_uploaded", False)):
                        existing_row["hf_uploaded"] = _try_hf_operation(
                            f"upload {stage_name} ({imgsz})",
                            upload_single_quantized_artifact_to_hf,
                            existing_artifact,
                            imgsz,
                            stage_name,
                            repo_id=repo_id,
                            token=token,
                            timeout_seconds=HF_UPLOAD_TIMEOUT_SECONDS,
                        )
                        _save_checkpoint(complete=False)
                    log.info("[SKIP-RESUME] imgsz=%d stage=%s", imgsz, stage_name)
                    continue
                if bgr_frames is None:
                    bgr_frames, nchw_tensors = _load_benchmark_frames(
                        Path(dataset_root),
                        eval_split,
                        imgsz,
                        max_frames=max(n_benchmark, 50),
                        manifest_path=manifest_path,
                        source_path=source_path,
                    )

                try:
                    artifact_path, stage_meta = _export_single_stage(
                        stage_name,
                        source_pt,
                        target_quant_dir,
                        calibration_images,
                        calibration_yaml,
                        imgsz=imgsz,
                        model_stem=model_stem,
                        calibrate_method=calibrate_method,
                        skip_existing=effective_skip_exports,
                    )
                    artifacts_by_res[imgsz][stage_name] = artifact_path
                    if "excluded_nodes" in stage_meta:
                        onnx_excluded_nodes[str(imgsz)] = list(stage_meta["excluded_nodes"])
                    if "onnx_fallback_full_head" in stage_meta:
                        onnx_fallback_flags[str(imgsz)] = bool(stage_meta["onnx_fallback_full_head"])

                    measured_row, measured_detail = _measure_stage_artifact(
                        artifact_path,
                        Path(dataset_root),
                        stage_name=stage_name,
                        imgsz=imgsz,
                        bgr_frames=bgr_frames,
                        nchw_tensors=nchw_tensors,
                        eval_split=eval_split,
                        n_warmup=n_warmup,
                        n_benchmark=n_benchmark,
                        collection_conf=collection_conf,
                        manifest_path=manifest_path,
                        source_path=source_path,
                        run_eval=(stage_name != "pytorch_fp32" or eval_pytorch),
                    )
                    rows_by_key[(imgsz, stage_name)] = measured_row
                    if measured_detail is not None:
                        per_stage_details[f"{stage_name}_{imgsz}"] = measured_detail
                except Exception as exc:
                    rows_by_key[(imgsz, stage_name)] = _failed_stage_row(imgsz, stage_name, str(exc))
                    _save_checkpoint(complete=False)
                else:
                    _save_checkpoint(complete=False)
                    if upload_to_hf and is_int8_stage:
                        rows_by_key[(imgsz, stage_name)]["hf_uploaded"] = _try_hf_operation(
                            f"upload {stage_name} ({imgsz})",
                            upload_single_quantized_artifact_to_hf,
                            artifact_path,
                            imgsz,
                            stage_name,
                            repo_id=repo_id,
                            token=token,
                            timeout_seconds=HF_UPLOAD_TIMEOUT_SECONDS,
                        )
                        _save_checkpoint(complete=False)
    finally:
        cleanup_calibration_dataset(target_working_dir, run_name=cal_run_name)

    results_df, failed_stages, csv_path, metadata_path = _save_checkpoint()

    if upload_to_hf:
        _try_hf_operation(
            "upload reports",
            upload_quantization_reports_to_hf,
            target_quant_dir,
            eval_split=eval_split,
            tag=hf_tag,
            repo_id=repo_id,
            token=token,
            timeout_seconds=HF_UPLOAD_TIMEOUT_SECONDS,
        )

    return {
        "results_df": results_df,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "artifacts": artifacts_by_res,
        "failed_stages": failed_stages,
    }




PTQ_VARIANT_SPECS: dict[str, dict[str, object]] = {
    "onnx_int8_minmax_head_int8": {
        "engine": "ort",
        "calibrate_method": "MinMax",
        "moving_average": False,
        "exclude_head_convs": False,
        "resolutions": (1280, 640),
    },
    "onnx_int8_moving_avg_head_int8": {
        "engine": "ort",
        "calibrate_method": "MinMax",
        "moving_average": True,
        "exclude_head_convs": False,
        "resolutions": (1280, 640),
    },
    "onnx_int8_entropy_head_int8": {
        "engine": "ort",
        "calibrate_method": "Entropy",
        "moving_average": False,
        "exclude_head_convs": False,
        "resolutions": (1280, 640),
    },
    "onnx_int8_percentile_head_int8": {
        "engine": "ort",
        "calibrate_method": "Percentile",
        "moving_average": False,
        "calibration_symmetric": False,
        "percentile": 99.99,
        "exclude_head_convs": False,
        "resolutions": (1280, 640),
    },
    "onnx_int8_minmax_head_fp32": {
        "engine": "ort",
        "calibrate_method": "MinMax",
        "moving_average": False,
        "exclude_head_convs": True,
        "resolutions": (1280, 640),
    },
    "onnx_int8_moving_avg_head_fp32": {
        "engine": "ort",
        "calibrate_method": "MinMax",
        "moving_average": True,
        "exclude_head_convs": True,
        "resolutions": (1280, 640),
    },
    "onnx_int8_entropy_head_fp32": {
        "engine": "ort",
        "calibrate_method": "Entropy",
        "moving_average": False,
        "exclude_head_convs": True,
        "resolutions": (1280, 640),
    },
    "onnx_int8_percentile_head_fp32": {
        "engine": "ort",
        "calibrate_method": "Percentile",
        "moving_average": False,
        "calibration_symmetric": False,
        "percentile": 99.99,
        "exclude_head_convs": True,
        "resolutions": (1280, 640),
    },
    "onnx_int8_nncf_head_int8": {
        "engine": "nncf_onnx",
        "exclude_head_convs": False,
        "fast_bias_correction": True,
        "resolutions": (1280, 640),
    },
    "onnx_int8_nncf_head_fp32": {
        "engine": "nncf_onnx",
        "exclude_head_convs": True,
        "fast_bias_correction": True,
        "resolutions": (1280, 640),
    },
    "openvino_int8_nncf_full_bias_head_fp32": {
        "engine": "nncf_openvino",
        "fast_bias_correction": False,
        "resolutions": (1280,),
    },
}

_PTQ_BASELINE_STAGE_ALIAS = {
    "onnx_fp32": "onnx_fp32",
    "onnx_int8": "onnx_int8_minmax_head_int8",
    "openvino_fp32": "openvino_fp32",
    "openvino_int8": "openvino_int8_ultralytics",
}

_PTQ_REPORT_STAGE_ORDER = (
    "onnx_fp32",
    *tuple(k for k in PTQ_VARIANT_SPECS if k.startswith("onnx_int8")),
    "openvino_fp32",
    "openvino_int8_ultralytics",
    *tuple(k for k in PTQ_VARIANT_SPECS if k.startswith("openvino_int8")),
)


def _seed_ptq_baseline_rows(
    quantization_dir: Path,
    eval_split: str,
    *,
    source_sha256: str,
    resolutions: tuple[int, ...],
    rows_by_key: dict[tuple[int, str], dict[str, object]],
    per_stage_details: dict[str, dict[str, object]],
) -> None:
    base_meta_path = Path(quantization_dir) / "reports" / eval_split / "quantization_metadata.json"
    if not base_meta_path.is_file():
        return
    try:
        base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if base_meta.get("source_weights_sha256") != source_sha256:
        return

    base_details = base_meta.get("stage_details") or {}
    for row in base_meta.get("summary_rows") or []:
        if row.get("status") != "ok":
            continue
        imgsz = int(row["imgsz"])
        orig_stage = str(row["stage"])
        mapped_stage = _PTQ_BASELINE_STAGE_ALIAS.get(orig_stage)
        if mapped_stage is None or imgsz not in resolutions:
            continue
        artifact = Path(str(row.get("artifact", "")))
        if not _is_valid_stage_artifact(mapped_stage, artifact):
            continue
        if (imgsz, mapped_stage) not in rows_by_key:
            cleaned = _clean_checkpoint_row(row)
            cleaned["stage"] = mapped_stage
            cleaned["hf_uploaded"] = False
            rows_by_key[(imgsz, mapped_stage)] = cleaned
        detail_key = f"{orig_stage}_{imgsz}"
        mapped_detail_key = f"{mapped_stage}_{imgsz}"
        if detail_key in base_details and mapped_detail_key not in per_stage_details:
            per_stage_details[mapped_detail_key] = base_details[detail_key]


def _ensure_fp32_for_ptq(
    fp32_stage: str,
    source_pt: Path,
    quantization_dir: Path,
    *,
    imgsz: int,
    model_stem: str,
) -> Path:
    paths = get_quantized_model_paths(quantization_dir, imgsz, model_stem=model_stem)
    if fp32_stage == "onnx_fp32":
        if _is_valid_stage_artifact("onnx_fp32", paths["onnx_fp32"]):
            return paths["onnx_fp32"]
        return export_onnx_fp32(source_pt, paths["onnx_fp32"], imgsz=imgsz)
    if fp32_stage == "openvino_fp32":
        if _is_valid_stage_artifact("openvino_fp32", paths["openvino_fp32"]):
            return paths["openvino_fp32"]
        return export_openvino_fp32(source_pt, paths["openvino_fp32"], imgsz=imgsz)
    raise ValueError(f"Unsupported FP32 stage: {fp32_stage!r}")


def _export_ptq_variant(
    variant_name: str,
    spec: dict[str, object],
    *,
    source_pt: Path,
    quantization_dir: Path,
    calibration_images: list[Path],
    imgsz: int,
    model_stem: str,
    skip_existing: bool,
) -> Path:
    ablation_dir = Path(quantization_dir) / "ptq_ablation" / str(imgsz)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    engine = str(spec["engine"])

    if engine == "nncf_openvino":
        ov_fp32 = _ensure_fp32_for_ptq(
            "openvino_fp32",
            source_pt,
            quantization_dir,
            imgsz=imgsz,
            model_stem=model_stem,
        )
        out_dir = ablation_dir / f"{model_stem}_{variant_name}_{imgsz}_openvino_model"
        if skip_existing and _is_valid_stage_artifact(variant_name, out_dir):
            return out_dir
        return export_openvino_int8_nncf(
            ov_fp32,
            out_dir,
            calibration_images,
            imgsz=imgsz,
            fast_bias_correction=bool(spec.get("fast_bias_correction", False)),
        )

    onnx_fp32 = _ensure_fp32_for_ptq(
        "onnx_fp32",
        source_pt,
        quantization_dir,
        imgsz=imgsz,
        model_stem=model_stem,
    )
    out_onnx = ablation_dir / f"{model_stem}_{variant_name}_{imgsz}.onnx"
    if skip_existing and _is_valid_stage_artifact(variant_name, out_onnx):
        return out_onnx

    if engine == "ort":
        return quantize_onnx_int8(
            onnx_fp32,
            out_onnx,
            calibration_images,
            imgsz=imgsz,
            calibrate_method=str(spec["calibrate_method"]),
            exclude_head_convs=bool(spec["exclude_head_convs"]),
            moving_average=bool(spec.get("moving_average", False)),
            calibration_symmetric=spec.get("calibration_symmetric"),
            percentile=spec.get("percentile"),
        )
    if engine == "nncf_onnx":
        return quantize_onnx_int8_nncf(
            onnx_fp32,
            out_onnx,
            calibration_images,
            imgsz=imgsz,
            exclude_head_convs=bool(spec["exclude_head_convs"]),
            fast_bias_correction=bool(spec.get("fast_bias_correction", True)),
        )
    raise ValueError(f"Unsupported PTQ engine: {engine!r}")


def run_ptq_variant_ablation(
    dataset_root: Path,
    *,
    weights_path: Path | None = None,
    quantization_dir: Path | None = None,
    working_dir: Path | None = None,
    resolutions: tuple[int, ...] = QUANTIZATION_RESOLUTIONS,
    variants: tuple[str, ...] = tuple(PTQ_VARIANT_SPECS),
    eval_split: str = "val",
    resume: bool = False,
    skip_existing_exports: bool = False,
    max_calibration_images: int = DEFAULT_CALIBRATION_IMAGES,
    seed: int = 42,
    n_warmup: int = 10,
    n_benchmark: int = 100,
    collection_conf: float = 0.01,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    model_stem: str = YOLO_MODEL_STEM,
) -> dict[str, object]:
    """Runs isolated local PTQ ablation across calibration/head/NNCF settings without touching HF."""
    if eval_split not in {"val", "test"}:
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")
    if not resolutions or any(r <= 0 for r in resolutions):
        raise ValueError(f"resolutions must be non-empty positive ints, got {resolutions!r}")
    if not variants or any(v not in PTQ_VARIANT_SPECS for v in variants):
        raise ValueError(f"Invalid variants {variants!r}; expected subset of {tuple(PTQ_VARIANT_SPECS)}")

    report_split = f"ptq_ablation_{eval_split}"
    target_quant_dir = Path(quantization_dir) if quantization_dir is not None else config.QUANTIZATION_DIR
    target_working_dir = Path(working_dir) if working_dir is not None else config.WORKING_DIR
    cache_dir = target_working_dir / "hf_cache" if working_dir is not None else config.LOCAL_HF_DATA_DIR
    source_pt = resolve_source_weights(
        weights_path,
        cache_dir=cache_dir if weights_path is None else None,
    )
    source_sha256 = _file_sha256(source_pt)

    rows_by_key: dict[tuple[int, str], dict[str, object]] = {}
    per_stage_details: dict[str, dict[str, object]] = {}
    if resume:
        rows_by_key, per_stage_details, _, _ = _load_resume_checkpoint(
            target_quant_dir,
            report_split,
            source_sha256=source_sha256,
            max_calibration_images=max_calibration_images,
            seed=seed,
            calibrate_method="ptq_ablation",
        )

    _seed_ptq_baseline_rows(
        target_quant_dir,
        eval_split,
        source_sha256=source_sha256,
        resolutions=resolutions,
        rows_by_key=rows_by_key,
        per_stage_details=per_stage_details,
    )

    effective_skip_exports = resume or skip_existing_exports
    cal_run_name = "calibration_ptq_ablation"
    _, calibration_images = prepare_calibration_dataset(
        Path(dataset_root),
        target_working_dir,
        manifest_path=manifest_path,
        source_path=source_path,
        max_images=max_calibration_images,
        seed=seed,
        run_name=cal_run_name,
        overwrite=True,
    )

    def _build_metadata(results_df: pd.DataFrame, *, complete: bool, failed_items: list[dict[str, object]]) -> dict[str, object]:
        return {
            "source_weights": str(source_pt),
            "source_weights_sha256": source_sha256,
            "eval_split": eval_split,
            "report_split": report_split,
            "resolutions": list(resolutions),
            "variants": list(variants),
            "collection_conf": collection_conf,
            "complete": complete,
            "failed_stages": failed_items,
            "benchmark": {
                "n_warmup": n_warmup,
                "n_benchmark": n_benchmark,
                "split": eval_split,
            },
            "calibration": {
                "manifest_path": str(manifest_path),
                "max_images": max_calibration_images,
                "actual_images": len(calibration_images),
                "seed": seed,
                "filenames": [path.name for path in calibration_images],
            },
            "onnx_quantization": {
                "format": "QDQ",
                "activation_type": "QUInt8",
                "weight_type": "QInt8",
                "per_channel": True,
                "reduce_range": False,
                "calibrate_method": "ptq_ablation",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "packages": _extract_runtime_versions(),
            },
            "stage_details": per_stage_details,
            "summary_rows": results_df.to_dict(orient="records"),
        }

    def _save_checkpoint(*, complete: bool | None = None) -> tuple[pd.DataFrame, list[dict[str, object]], Path, Path]:
        snapshot = _ordered_rows_snapshot(rows_by_key, stage_order=_PTQ_REPORT_STAGE_ORDER)
        snapshot_df = _annotate_ablation_deltas(snapshot)
        failed_items = [
            {"imgsz": int(r["imgsz"]), "stage": str(r["stage"]), "error": str(r.get("error", ""))}
            for r in snapshot
            if r.get("status") != "ok"
        ]
        is_complete = (len(failed_items) == 0) if complete is None else complete
        saved_csv, saved_meta = save_quantization_reports(
            snapshot_df,
            target_quant_dir,
            _build_metadata(snapshot_df, complete=is_complete, failed_items=failed_items),
            eval_split=report_split,
        )
        return snapshot_df, failed_items, saved_csv, saved_meta

    try:
        for imgsz in resolutions:
            bgr_frames = None
            nchw_tensors = None

            active_variants = [
                v for v in variants if imgsz in tuple(PTQ_VARIANT_SPECS[v]["resolutions"])
            ]
            if not active_variants:
                continue

            needed_baselines = []
            if any(v.startswith("onnx_int8") for v in active_variants):
                needed_baselines.append("onnx_fp32")
            if any(v.startswith("openvino_int8") for v in active_variants):
                needed_baselines.append("openvino_fp32")

            for base_stage in needed_baselines:
                if (imgsz, base_stage) in rows_by_key:
                    continue
                if bgr_frames is None:
                    bgr_frames, nchw_tensors = _load_benchmark_frames(
                        Path(dataset_root),
                        eval_split,
                        imgsz,
                        max_frames=max(n_benchmark, 50),
                        manifest_path=manifest_path,
                        source_path=source_path,
                    )
                try:
                    base_artifact = _ensure_fp32_for_ptq(
                        base_stage,
                        source_pt,
                        target_quant_dir,
                        imgsz=imgsz,
                        model_stem=model_stem,
                    )
                    row, detail = _measure_stage_artifact(
                        base_artifact,
                        Path(dataset_root),
                        stage_name=base_stage,
                        imgsz=imgsz,
                        bgr_frames=bgr_frames,
                        nchw_tensors=nchw_tensors,
                        eval_split=eval_split,
                        n_warmup=n_warmup,
                        n_benchmark=n_benchmark,
                        collection_conf=collection_conf,
                        manifest_path=manifest_path,
                        source_path=source_path,
                        run_eval=True,
                    )
                    rows_by_key[(imgsz, base_stage)] = row
                    if detail is not None:
                        per_stage_details[f"{base_stage}_{imgsz}"] = detail
                except Exception as exc:
                    rows_by_key[(imgsz, base_stage)] = _failed_stage_row(imgsz, base_stage, str(exc))
                _save_checkpoint(complete=False)

            for variant_name in active_variants:
                if (imgsz, variant_name) in rows_by_key:
                    log.info("[SKIP-RESUME] imgsz=%d variant=%s", imgsz, variant_name)
                    continue

                if bgr_frames is None:
                    bgr_frames, nchw_tensors = _load_benchmark_frames(
                        Path(dataset_root),
                        eval_split,
                        imgsz,
                        max_frames=max(n_benchmark, 50),
                        manifest_path=manifest_path,
                        source_path=source_path,
                    )

                try:
                    artifact_path = _export_ptq_variant(
                        variant_name,
                        PTQ_VARIANT_SPECS[variant_name],
                        source_pt=source_pt,
                        quantization_dir=target_quant_dir,
                        calibration_images=calibration_images,
                        imgsz=imgsz,
                        model_stem=model_stem,
                        skip_existing=effective_skip_exports,
                    )
                    row, detail = _measure_stage_artifact(
                        artifact_path,
                        Path(dataset_root),
                        stage_name=variant_name,
                        imgsz=imgsz,
                        bgr_frames=bgr_frames,
                        nchw_tensors=nchw_tensors,
                        eval_split=eval_split,
                        n_warmup=n_warmup,
                        n_benchmark=n_benchmark,
                        collection_conf=collection_conf,
                        manifest_path=manifest_path,
                        source_path=source_path,
                        run_eval=True,
                    )
                    rows_by_key[(imgsz, variant_name)] = row
                    if detail is not None:
                        per_stage_details[f"{variant_name}_{imgsz}"] = detail
                except Exception as exc:
                    rows_by_key[(imgsz, variant_name)] = _failed_stage_row(imgsz, variant_name, str(exc))
                _save_checkpoint(complete=False)
    finally:
        cleanup_calibration_dataset(target_working_dir, run_name=cal_run_name)

    results_df, failed_stages, csv_path, metadata_path = _save_checkpoint()
    return {
        "results_df": results_df,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "failed_stages": failed_stages,
    }