from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _load_quantization(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EXTRACTED_DATA_PATH", str(tmp_path))

    from src.processing import quantization

    monkeypatch.setattr(quantization, "REPO_ID", "repo/test")
    monkeypatch.setattr(quantization, "HF_TOKEN", "token-test")
    monkeypatch.setattr(quantization, "benchmark_model", lambda *a, **k: {"label": a[1], "fps": 10.0, "mean_ms": 100.0})
    monkeypatch.setattr(quantization, "validate_model_accuracy", lambda *a, **k: 0.9)
    return quantization


def test_export_pipeline_propagates_trt_export_failure(tmp_path, monkeypatch):
    quantization = _load_quantization(tmp_path, monkeypatch)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")

    fp32_model = Mock()
    fp32_model.export.side_effect = RuntimeError("trt export failed")
    yolo_instance = Mock()
    yolo_instance.to.return_value = fp32_model
    monkeypatch.setattr(quantization, "YOLO", Mock(return_value=yolo_instance))

    with pytest.raises(RuntimeError, match="trt export failed"):
        quantization.run_export_and_benchmark_pipeline(weights, tmp_path / "data.yaml", "model", imgsz=64)


def test_export_pipeline_raises_when_engine_file_missing(tmp_path, monkeypatch):
    quantization = _load_quantization(tmp_path, monkeypatch)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")

    fp32_model = Mock()
    fp32_model.export.return_value = str(tmp_path / "missing.engine")
    yolo_instance = Mock()
    yolo_instance.to.return_value = fp32_model
    monkeypatch.setattr(quantization, "YOLO", Mock(return_value=yolo_instance))

    with pytest.raises(FileNotFoundError, match="Engine|engine"):
        quantization.run_export_and_benchmark_pipeline(weights, tmp_path / "data.yaml", "model", imgsz=64)


def test_export_pipeline_propagates_hf_upload_failure(tmp_path, monkeypatch):
    quantization = _load_quantization(tmp_path, monkeypatch)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    engine = tmp_path / "best.engine"
    engine.write_bytes(b"engine")

    fp32_model = Mock()
    fp32_model.export.return_value = str(engine)
    yolo_instance = Mock()
    yolo_instance.to.return_value = fp32_model
    monkeypatch.setattr(quantization, "YOLO", Mock(return_value=yolo_instance))

    api = SimpleNamespace(upload_file=Mock(side_effect=RuntimeError("hf upload failed")))
    monkeypatch.setattr(quantization, "HfApi", Mock(return_value=api))

    with pytest.raises(RuntimeError, match="hf upload failed"):
        quantization.run_export_and_benchmark_pipeline(weights, tmp_path / "data.yaml", "model", imgsz=64)