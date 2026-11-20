from collections import Counter, deque
import logging
import math
import os
import random
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import onnx
from ultralytics import YOLO
import yaml

from src.config import (
    HF_MODEL_PATH_PT,
    HF_TOKEN,
    MANIFESTS_DIR,
    REPO_ID,
    YOLO_MODEL_STEM,
    get_quantized_model_paths,
)
from src.processing.dataset import parse_frame_filename_logic
from src.processing.diagnostics import letterbox_image
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


def select_onnx_nodes_to_exclude(model_proto: onnx.ModelProto) -> list[str]:
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
        elif in_head and node.op_type in _SENSITIVE_HEAD_OP_TYPES:
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


def quantize_onnx_int8(
    onnx_fp32_path: Path,
    output_path: Path,
    calibration_images: list[Path],
    *,
    imgsz: int,
    calibrate_method: str = "MinMax",
) -> Path:
    """Quantizes an ONNX FP32 model to QDQ INT8 with end2end head exclusion and ARM64 parity."""
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

    source_model = onnx.load(str(onnx_fp32_path))
    if not source_model.graph.input:
        raise ValueError(f"ONNX model has no graph inputs: {onnx_fp32_path}")
    input_name = source_model.graph.input[0].name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as tmp_dir:
        tmp_root = Path(tmp_dir)
        pre_path = _preprocess_onnx_for_quantization(onnx_fp32_path, tmp_root / "pre.onnx")
        pre_model = onnx.load(str(pre_path))
        nodes_to_exclude = select_onnx_nodes_to_exclude(pre_model)
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
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
        if not tmp_int8.is_file():
            raise FileNotFoundError(f"Quantized ONNX INT8 output was not created: {tmp_int8}")

        _copy_onnx_metadata_and_ir(source_model, tmp_int8)
        _move_artifact(tmp_int8, output_path)

    log.info(
        "Exported ONNX INT8 (%s, excluded %d head nodes) -> %s",
        calibrate_method,
        len(nodes_to_exclude),
        output_path,
    )
    return output_path


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