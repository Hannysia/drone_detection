import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest
import yaml

from src.processing import quantization
from src.processing.manifests import CATEGORY_DIRS, write_manifest


def _write_calibration_bundle(tmp_path, rows):
    root = tmp_path / "curated"
    for row in rows:
        folder = root / row["split"] / CATEGORY_DIRS[row["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        image = folder / row["filename"]
        assert cv2.imwrite(str(image), np.full((32, 48, 3), 120, dtype=np.uint8))
        if row["category"] == "positive":
            image.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    manifest_path = tmp_path / "main.csv"
    write_manifest(rows, manifest_path)
    return root, manifest_path


def _make_conv_sigmoid_onnx(path: Path, *, ir_version: int = 9) -> Path:
    rng = np.random.default_rng(7)
    w_init = helper.make_tensor(
        "conv_w",
        TensorProto.FLOAT,
        [4, 3, 3, 3],
        rng.normal(0.0, 0.2, size=(4, 3, 3, 3)).astype(np.float32).flatten().tolist(),
    )
    bias_init = helper.make_tensor(
        "add_b",
        TensorProto.FLOAT,
        [1, 4, 1, 1],
        [0.1, -0.1, 0.2, -0.2],
    )
    nodes = [
        helper.make_node(
            "Conv",
            ["images", "conv_w"],
            ["conv_out"],
            name="/model.0/conv/Conv",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="/model.0/act/Relu"),
        helper.make_node("Add", ["relu_out", "add_b"], ["head_feat"], name="/model.23/Add"),
        helper.make_node("Sigmoid", ["head_feat"], ["output0"], name="/model.23/Sigmoid"),
    ]
    graph = helper.make_graph(
        nodes,
        "yolo_head_test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, 32, 32])],
        initializer=[w_init, bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = ir_version
    for key, value in [("task", "detect"), ("end2end", "True"), ("stride", "32")]:
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, str(path))
    return path


def test_verify_hf_access_and_resolve_source_weights(tmp_path, monkeypatch):
    mock_ready = Mock(return_value=("user/ds", "tok"))
    monkeypatch.setattr(quantization, "assert_hf_ready", mock_ready)

    assert quantization.verify_hf_access("user/ds", "tok") == ("user/ds", "tok")
    mock_ready.assert_called_once_with(repo_id="user/ds", token="tok")

    local_pt = tmp_path / "weights.pt"
    local_pt.write_bytes(b"pt")
    assert quantization.resolve_source_weights(local_pt) == local_pt

    with pytest.raises(FileNotFoundError, match="not found"):
        quantization.resolve_source_weights(tmp_path / "missing.pt")

    cached_pt = tmp_path / "models" / "yolo26n_run_with_negatives_best.pt"
    cached_pt.parent.mkdir(parents=True)
    cached_pt.write_bytes(b"cached")
    assert quantization.resolve_source_weights(cache_dir=tmp_path) == cached_pt

    dl_dir = tmp_path / "dl"
    dl_target = dl_dir / "models" / "yolo26n_run_with_negatives_best.pt"
    dl_target.parent.mkdir(parents=True)
    dl_target.write_bytes(b"downloaded")
    mock_dl = Mock(return_value=str(dl_target))
    monkeypatch.setattr(quantization, "hf_hub_download", mock_dl)

    resolved = quantization.resolve_source_weights(cache_dir=dl_dir, repo_id="user/ds", token="tok")
    assert resolved == dl_target


def test_get_artifact_size_mb_handles_file_and_openvino_dir(tmp_path):
    pt_file = tmp_path / "model.pt"
    pt_file.write_bytes(b"x" * (1024 * 1024))
    assert quantization.get_artifact_size_mb(pt_file) == pytest.approx(1.0)

    ov_dir = tmp_path / "model_openvino_model"
    ov_dir.mkdir()
    (ov_dir / "model.xml").write_bytes(b"a" * (512 * 1024))
    (ov_dir / "model.bin").write_bytes(b"b" * (512 * 1024))
    assert quantization.get_artifact_size_mb(ov_dir) == pytest.approx(1.0)

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        quantization.get_artifact_size_mb(empty_dir)

    with pytest.raises(FileNotFoundError):
        quantization.get_artifact_size_mb(tmp_path / "missing")


def test_prepare_calibration_dataset_stratifies_train_only_across_videos(tmp_path):
    rows = [
        {"filename": f"quad_rgb_sky_001_f{i:04d}.jpg", "split": "train", "category": "positive", "excluded_at": ""}
        for i in range(1, 11)
    ] + [
        {"filename": "quad_rgb_sky_002_f0001.jpg", "split": "train", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_001_f0011.jpg", "split": "train", "category": "negative", "excluded_at": ""},
        {"filename": "wing_thermal_forest_003_f0001.jpg", "split": "train", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_099_f0001.jpg", "split": "train", "category": "positive", "excluded_at": "embeddings"},
        {"filename": "quad_rgb_sky_900_f0001.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_901_f0001.jpg", "split": "test", "category": "negative", "excluded_at": ""},
    ]
    root, manifest_path = _write_calibration_bundle(tmp_path, rows)
    working_dir = tmp_path / "working"

    yaml_path, cal_images = quantization.prepare_calibration_dataset(
        root,
        working_dir,
        manifest_path=manifest_path,
        max_images=6,
        seed=42,
    )

    assert yaml_path.is_file()
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert cfg["train"] == "images/cal"
    assert cfg["val"] == "images/cal"

    names = {path.name for path in cal_images}
    assert len(names) == 6
    assert "quad_rgb_sky_002_f0001.jpg" in names
    assert "wing_thermal_forest_003_f0001.jpg" in names
    assert "quad_rgb_sky_099_f0001.jpg" not in names
    assert "quad_rgb_sky_900_f0001.jpg" not in names
    assert "quad_rgb_sky_901_f0001.jpg" not in names

    pos_names = {
        p.name
        for p in cal_images
        if (yaml_path.parent / "yolo_dataset_calibration_ptq" / "labels" / "cal" / f"{p.stem}.txt").exists()
    }
    assert len(pos_names) == 5

    quantization.cleanup_calibration_dataset(working_dir)
    assert not yaml_path.exists()


def test_select_onnx_nodes_to_exclude_protects_dfl_head_tail_and_topk():
    nodes = [
        helper.make_node("Conv", ["x"], ["c0"], name="/model.0/conv/Conv"),
        helper.make_node("Conv", ["c0"], ["h_box"], name="/model.23/one2one_cv2.0/Conv"),
        helper.make_node("Conv", ["h_box"], ["dfl_c"], name="/model.23/dfl/conv/Conv"),
        helper.make_node("Softmax", ["dfl_c"], ["dfl_s"], name="/model.23/dfl/Softmax"),
        helper.make_node("Add", ["dfl_s"], ["decoded"], name="/model.23/Add"),
        helper.make_node("Sigmoid", ["decoded"], ["probs"], name="/model.23/Sigmoid"),
        helper.make_node("TopK", ["probs"], ["top_vals", "top_idx"], name="/model.23/TopK"),
        helper.make_node("Mod", ["top_idx"], ["cls_idx"], name="/model.23/Mod"),
    ]
    graph = helper.make_graph(nodes, "yolo26_end2end", [], [])
    model = helper.make_model(graph)

    excluded = quantization.select_onnx_nodes_to_exclude(model)

    assert "/model.0/conv/Conv" not in excluded
    assert "/model.23/one2one_cv2.0/Conv" not in excluded
    assert excluded == [
        "/model.23/dfl/conv/Conv",
        "/model.23/dfl/Softmax",
        "/model.23/Add",
        "/model.23/Sigmoid",
        "/model.23/TopK",
        "/model.23/Mod",
    ]


def test_quantize_onnx_int8_end_to_end_preserves_metadata_and_inserts_qdq(tmp_path):
    fp32_onnx = _make_conv_sigmoid_onnx(tmp_path / "yolo_fp32.onnx", ir_version=9)
    cal_img = tmp_path / "quad_rgb_sky_001_f0001.jpg"
    assert cv2.imwrite(str(cal_img), np.random.default_rng(1).integers(0, 255, (40, 60, 3), dtype=np.uint8))

    out_int8 = tmp_path / "yolo_int8.onnx"
    result_path = quantization.quantize_onnx_int8(
        fp32_onnx,
        out_int8,
        [cal_img],
        imgsz=32,
    )

    assert result_path == out_int8
    assert out_int8.is_file()

    quantized_model = onnx.load(str(out_int8))
    assert quantized_model.ir_version <= quantization.MAX_ONNX_IR_VERSION
    meta = {prop.key: prop.value for prop in quantized_model.metadata_props}
    assert meta["task"] == "detect"
    assert meta["end2end"] == "True"
    op_types = {node.op_type for node in quantized_model.graph.node}
    assert "QuantizeLinear" in op_types
    assert "DequantizeLinear" in op_types

    session = ort.InferenceSession(str(out_int8), providers=["CPUExecutionProvider"])
    out = session.run(None, {"images": np.zeros((1, 3, 32, 32), dtype=np.float32)})[0]
    assert out.shape == (1, 4, 32, 32)


def test_export_stage_models_isolates_resolutions_and_propagates_failures(tmp_path, monkeypatch):
    weights = tmp_path / "yolo26n_run_with_negatives_best.pt"
    weights.write_bytes(b"pt")
    cal_img = tmp_path / "cal.jpg"
    assert cv2.imwrite(str(cal_img), np.zeros((32, 32, 3), dtype=np.uint8))
    cal_yaml = tmp_path / "cal.yaml"
    cal_yaml.write_text("path: .\n", encoding="utf-8")

    export_calls = []

    def make_yolo(path):
        staged_pt = Path(path)

        def fake_export(**kwargs):
            export_calls.append(dict(kwargs))
            if kwargs["format"] == "onnx":
                out = staged_pt.with_suffix(".onnx")
                _make_conv_sigmoid_onnx(out)
                return str(out)
            suffix = "_int8_openvino_model" if kwargs["int8"] else "_openvino_model"
            out_dir = staged_pt.parent / f"{staged_pt.stem}{suffix}"
            out_dir.mkdir()
            (out_dir / f"{staged_pt.stem}.xml").write_text("<net/>", encoding="utf-8")
            (out_dir / f"{staged_pt.stem}.bin").write_bytes(b"bin")
            return str(out_dir)

        return SimpleNamespace(export=fake_export)

    def fake_qdq(src, dst, *args, **kwargs):
        dst.write_bytes(b"qdq")
        return dst

    monkeypatch.setattr(quantization, "YOLO", make_yolo)
    monkeypatch.setattr(quantization, "quantize_onnx_int8", fake_qdq)

    quant_dir = tmp_path / "quantization"
    stages_1280 = quantization.export_stage_models(weights, quant_dir, [cal_img], cal_yaml, imgsz=1280)
    stages_640 = quantization.export_stage_models(weights, quant_dir, [cal_img], cal_yaml, imgsz=640)

    assert stages_1280["onnx_fp32"].parent.name == "1280"
    assert stages_640["onnx_fp32"].parent.name == "640"
    assert stages_1280["openvino_fp32"].is_dir()
    assert stages_1280["openvino_int8"].is_dir()
    assert set(stages_1280.values()).isdisjoint(set(stages_640.values()))

    ov_int8_calls = [call for call in export_calls if call["format"] == "openvino" and call["int8"]]
    assert [call["imgsz"] for call in ov_int8_calls] == [1280, 640]
    assert all(
        call["data"] == str(cal_yaml)
        and call["device"] == "cpu"
        and call["batch"] == 1
        and call["dynamic"] is False
        and call["half"] is False
        for call in ov_int8_calls
    )

    failing = SimpleNamespace(export=Mock(return_value=str(tmp_path / "missing_openvino_model")))
    monkeypatch.setattr(quantization, "YOLO", Mock(return_value=failing))
    with pytest.raises(FileNotFoundError, match="OpenVINO"):
        quantization.export_openvino_int8(weights, tmp_path / "ov_out", cal_yaml, imgsz=640)


def test_benchmark_stage_model_measures_real_onnx_and_mocked_yolo(tmp_path, monkeypatch):
    fp32_onnx = _make_conv_sigmoid_onnx(tmp_path / "model.onnx", ir_version=9)
    bgr = np.full((32, 48, 3), 128, dtype=np.uint8)
    nchw = quantization._bgr_to_nchw(bgr, imgsz=32)

    fake_yolo = Mock()
    fake_yolo.predict.return_value = [Mock(boxes=None)]
    monkeypatch.setattr(quantization, "_load_yolo_for_artifact", Mock(return_value=fake_yolo))

    stats = quantization.benchmark_stage_model(
        fp32_onnx,
        [bgr],
        [nchw],
        imgsz=32,
        n_warmup=1,
        n_benchmark=3,
    )

    assert stats["compute_mean_ms"] > 0.0
    assert stats["compute_p50_ms"] <= stats["compute_p95_ms"] <= stats["compute_p99_ms"]
    assert stats["e2e_mean_ms"] > 0.0
    assert stats["e2e_fps"] > 0.0
    assert fake_yolo.predict.call_count == 4
    assert fake_yolo.predict.call_args.kwargs["device"] == "cpu"
    assert fake_yolo.predict.call_args.kwargs["rect"] is False

    with pytest.raises(ValueError, match="non-empty"):
        quantization.benchmark_stage_model(fp32_onnx, [], [], imgsz=32)


def test_annotate_ablation_deltas_pairs_fp32_to_int8_within_each_resolution():
    rows = [
        {
            "imgsz": 1280,
            "stage": "pytorch_fp32",
            "size_mb": 5.5,
            "compute_mean_ms": 100.0,
            "e2e_mean_ms": 120.0,
            "mAP50": float("nan"),
            "mAP50-95": float("nan"),
            "AP50": float("nan"),
            "AP50_small": float("nan"),
            "Neg FPR @ R>=0.85 (%)": float("nan"),
        },
        {
            "imgsz": 1280,
            "stage": "onnx_fp32",
            "size_mb": 10.0,
            "compute_mean_ms": 80.0,
            "e2e_mean_ms": 96.0,
            "mAP50": 0.90,
            "mAP50-95": 0.75,
            "AP50": 0.90,
            "AP50_small": 0.70,
            "Neg FPR @ R>=0.85 (%)": 6.0,
        },
        {
            "imgsz": 1280,
            "stage": "onnx_int8",
            "size_mb": 3.0,
            "compute_mean_ms": 40.0,
            "e2e_mean_ms": 48.0,
            "mAP50": 0.89,
            "mAP50-95": 0.73,
            "AP50": 0.88,
            "AP50_small": 0.67,
            "Neg FPR @ R>=0.85 (%)": 8.0,
        },
        {
            "imgsz": 640,
            "stage": "onnx_fp32",
            "size_mb": 10.0,
            "compute_mean_ms": 20.0,
            "e2e_mean_ms": 30.0,
            "mAP50": 0.80,
            "mAP50-95": 0.60,
            "AP50": 0.80,
            "AP50_small": 0.50,
            "Neg FPR @ R>=0.85 (%)": 10.0,
        },
        {
            "imgsz": 640,
            "stage": "onnx_int8",
            "size_mb": 3.0,
            "compute_mean_ms": 10.0,
            "e2e_mean_ms": 15.0,
            "mAP50": 0.79,
            "mAP50-95": 0.59,
            "AP50": 0.78,
            "AP50_small": 0.48,
            "Neg FPR @ R>=0.85 (%)": 12.0,
        },
    ]

    df = quantization._annotate_ablation_deltas(rows).set_index(["imgsz", "stage"])

    onnx_1280 = df.loc[(1280, "onnx_int8")]
    onnx_640 = df.loc[(640, "onnx_int8")]
    assert onnx_1280["ap50_delta_vs_fp32"] == pytest.approx(-0.02)
    assert onnx_640["ap50_delta_vs_fp32"] == pytest.approx(-0.02)
    assert onnx_1280["speedup_compute_vs_fp32"] == pytest.approx(2.0)
    assert onnx_1280["speedup_compute_vs_pt"] == pytest.approx(2.5)
    assert onnx_1280["size_reduction_pct_vs_fp32"] == pytest.approx(70.0)


def test_run_cpu_quantization_ablation_uses_val_skips_pt_eval_and_cleans_up(tmp_path, monkeypatch):
    rows = [
        {"filename": "quad_rgb_sky_001_f0001.jpg", "split": "train", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_001_f0002.jpg", "split": "train", "category": "negative", "excluded_at": ""},
        {"filename": "quad_rgb_sky_002_f0001.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_002_f0002.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "quad_rgb_sky_003_f0001.jpg", "split": "test", "category": "positive", "excluded_at": ""},
    ]
    root, manifest_path = _write_calibration_bundle(tmp_path, rows)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    quant_dir = tmp_path / "quant"
    working_dir = tmp_path / "work"

    def fake_export_single(stage_name, pt, qdir, cal_imgs, cal_yaml, *, imgsz, model_stem, calibrate_method, skip_existing=False):
        res_dir = Path(qdir) / str(imgsz)
        res_dir.mkdir(parents=True, exist_ok=True)
        if stage_name == "pytorch_fp32":
            out = res_dir / f"{model_stem}.pt"
            out.write_bytes(b"pt")
            return out, {}
        if stage_name in {"onnx_fp32", "onnx_int8"}:
            suffix = "fp32" if stage_name == "onnx_fp32" else "int8"
            out = _make_conv_sigmoid_onnx(res_dir / f"{model_stem}_{suffix}_{imgsz}.onnx")
            meta = {"excluded_nodes": ["/model.23/Add"], "onnx_fallback_full_head": False} if stage_name == "onnx_int8" else {}
            return out, meta
        suffix = "fp32" if stage_name == "openvino_fp32" else "int8"
        ov_dir = res_dir / f"{model_stem}_{suffix}_{imgsz}_openvino_model"
        ov_dir.mkdir(exist_ok=True)
        (ov_dir / "model.xml").write_text("<net/>", encoding="utf-8")
        (ov_dir / "model.bin").write_bytes(b"bin")
        return ov_dir, {}

    eval_calls = []

    def fake_eval_stage(artifact, d_root, yaml_p, stage, *, imgsz, eval_split, **kwargs):
        eval_calls.append((imgsz, stage, eval_split))
        return {
            "metrics": {
                "mAP50": 0.90,
                "mAP50-95": 0.75,
                "AP50": 0.89,
                "AP50_small": 0.65,
                "AP50_medium": 0.85,
                "AP50_large": 0.92,
                "Recall_small @ Neg FPR<=5%": 0.55,
                "Neg FPR @ R>=0.85 (%)": 5.0,
                "Mean Neg FA Rate (%)": 4.0,
                "Max-F1 P": 0.92,
                "Max-F1 R": 0.88,
                "Max-F1 Recall_small": 0.60,
                "Max-F1 Neg FA Rate (%)": 3.0,
                "Recall_small (raw)": 0.70,
                "Max Recall (raw)": 0.95,
                "tp_conf_p10": 0.45,
                "tp_conf_median": 0.82,
                "tp_conf_p90": 0.95,
                "tp_recall_ge_0_30": 0.92,
            },
            "by_scale": [{"Scale": "small", "GT": 1, "AP50": 0.65}],
            "curves": {"thresholds": [0.1, 0.5], "precision": [0.9, 0.95]},
        }

    monkeypatch.setattr(quantization, "_export_single_stage", fake_export_single)
    monkeypatch.setattr(
        quantization,
        "benchmark_stage_model",
        lambda *a, **k: {
            "compute_mean_ms": 10.0,
            "compute_std_ms": 1.0,
            "compute_p50_ms": 10.0,
            "compute_p95_ms": 11.0,
            "compute_p99_ms": 12.0,
            "compute_fps": 100.0,
            "compute_fps_p50": 100.0,
            "e2e_mean_ms": 15.0,
            "e2e_std_ms": 1.0,
            "e2e_p50_ms": 15.0,
            "e2e_p95_ms": 16.0,
            "e2e_p99_ms": 17.0,
            "e2e_fps": 66.7,
            "e2e_fps_p50": 66.7,
            "rss_mb": 200.0,
            "peak_rss_mb": 220.0,
        },
    )
    monkeypatch.setattr(quantization, "evaluate_quantized_stage", fake_eval_stage)

    result = quantization.run_cpu_quantization_ablation(
        root,
        weights_path=weights,
        quantization_dir=quant_dir,
        working_dir=working_dir,
        resolutions=(1280, 640),
        n_warmup=1,
        n_benchmark=2,
        manifest_path=manifest_path,
    )

    assert len(result["results_df"]) == 10
    assert len(eval_calls) == 8
    assert all(stage != "pytorch_fp32" and split == "val" for _, stage, split in eval_calls)
    assert result["csv_path"] == quant_dir / "reports" / "val" / "ablation_results.csv"
    assert result["csv_path"].is_file()
    meta = json.loads(result["metadata_path"].read_text(encoding="utf-8"))
    assert meta["eval_split"] == "val"
    assert meta["eval_pytorch"] is False
    assert all("002_" not in name and "003_" not in name for name in meta["calibration"]["filenames"])
    assert not (working_dir / "dataset_calibration_ptq.yaml").exists()
    assert not (working_dir / "dataset_quantization_eval.yaml").exists()


def test_upload_quantization_artifacts_to_hf_uploads_only_int8_and_reports(tmp_path, monkeypatch):
    quant_dir = tmp_path / "quantization"
    report_dir = quant_dir / "reports" / "test"
    report_dir.mkdir(parents=True)
    (report_dir / "ablation_results.csv").write_text("stage\n", encoding="utf-8")
    (report_dir / "quantization_metadata.json").write_text("{}", encoding="utf-8")

    for imgsz in (1280, 640):
        res_dir = quant_dir / str(imgsz)
        res_dir.mkdir()
        (res_dir / f"yolo26n_run_with_negatives_best_int8_{imgsz}.onnx").write_bytes(b"int8")
        ov_int8 = res_dir / f"yolo26n_run_with_negatives_best_int8_{imgsz}_openvino_model"
        ov_int8.mkdir()
        (ov_int8 / "model.xml").write_text("<net/>", encoding="utf-8")
        (ov_int8 / "model.bin").write_bytes(b"bin")

    monkeypatch.setattr(quantization, "verify_hf_access", Mock(return_value=("user/repo", "tok")))
    api = SimpleNamespace(
        upload_folder=Mock(),
        create_tag=Mock(),
    )
    monkeypatch.setattr(quantization, "HfApi", Mock(return_value=api))

    quantization.upload_quantization_artifacts_to_hf(
        quant_dir,
        eval_split="test",
        tag="cpu-quant-v1",
    )

    upload_kwargs = api.upload_folder.call_args.kwargs
    assert upload_kwargs["path_in_repo"] == "models/quantization"
    assert upload_kwargs["allow_patterns"] == [
        "reports/test/**",
        "1280/yolo26n_run_with_negatives_best_int8_1280.onnx",
        "1280/yolo26n_run_with_negatives_best_int8_1280_openvino_model/**",
        "640/yolo26n_run_with_negatives_best_int8_640.onnx",
        "640/yolo26n_run_with_negatives_best_int8_640_openvino_model/**",
    ]
    api.create_tag.assert_called_once_with(
        repo_id="user/repo",
        tag="cpu-quant-v1",
        repo_type="dataset",
    )

    api.upload_folder.side_effect = RuntimeError("hf upload failed")
    with pytest.raises(RuntimeError, match="hf upload failed"):
        quantization.upload_quantization_artifacts_to_hf(quant_dir, eval_split="test")


def _load_quantize_cli_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "06_quantize_cpu.py"
    spec = importlib.util.spec_from_file_location("quantize_cpu_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_quantized_stage_runs_single_pass_threshold_free_cpu_eval(tmp_path, monkeypatch):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    raw_db = [
        {
            "filename": "quad_rgb_sky_001_f0001.jpg",
            "is_positive": True,
            "width": 1000,
            "height": 1000,
            "gt_boxes": [[10.0, 10.0, 20.0, 20.0]],
            "pred_boxes": [[10.0, 10.0, 20.0, 20.0, 0.85]],
        },
        {
            "filename": "quad_rgb_sky_001_f0002.jpg",
            "is_positive": True,
            "width": 1000,
            "height": 1000,
            "gt_boxes": [[100.0, 100.0, 220.0, 220.0]],
            "pred_boxes": [[100.0, 100.0, 220.0, 220.0, 0.20]],
        },
        {
            "filename": "quad_rgb_sky_001_f0003.jpg",
            "is_positive": False,
            "width": 1000,
            "height": 1000,
            "gt_boxes": [],
            "pred_boxes": [[0.0, 0.0, 10.0, 10.0, 0.15], [0.0, 0.0, 10.0, 10.0, 0.005]],
        },
    ]
    mock_preds_db = Mock(return_value=raw_db)
    fake_model = Mock()

    monkeypatch.setattr(quantization, "_load_yolo_for_artifact", Mock(return_value=fake_model))
    monkeypatch.setattr(quantization, "run_dataset_predictions_db", mock_preds_db)

    with pytest.raises(ValueError, match="eval_split"):
        quantization.evaluate_quantized_stage(onnx_path, tmp_path, None, "onnx_int8", imgsz=640, eval_split="train")

    report = quantization.evaluate_quantized_stage(
        onnx_path,
        tmp_path,
        None,
        "onnx_int8",
        imgsz=640,
        eval_split="val",
        collection_conf=0.01,
    )

    mock_preds_db.assert_called_once()
    assert mock_preds_db.call_args.kwargs["device"] == "cpu"
    assert mock_preds_db.call_args.kwargs["collection_conf"] == pytest.approx(0.001)
    assert report["metrics"]["mAP50"] == pytest.approx(1.0)
    assert report["metrics"]["mAP50-95"] == pytest.approx(1.0)
    assert report["metrics"]["AP50"] == pytest.approx(1.0)
    assert report["metrics"]["AP50_small"] == pytest.approx(1.0)
    assert report["metrics"]["tp_conf_median"] == pytest.approx(0.525)
    assert report["metrics"]["tp_recall_ge_0_30"] == pytest.approx(0.5)
    assert len(report["by_scale"]) == 3
    assert len(report["curves"]["thresholds"]) > 0


def test_quantize_cpu_cli_defaults_and_flags(tmp_path, monkeypatch):
    cli = _load_quantize_cli_module()
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))

    import pandas as pd

    fake_result = {
        "results_df": pd.DataFrame([{"imgsz": 640, "stage": "onnx_int8", "size_mb": 3.0}]),
        "csv_path": tmp_path / "ablation.csv",
        "metadata_path": tmp_path / "meta.json",
    }
    mock_run = Mock(return_value=fake_result)
    monkeypatch.setattr(cli, "run_cpu_quantization_ablation", mock_run)

    assert cli.main([]) == 0
    first_args, first_kwargs = mock_run.call_args
    assert first_args[0] == tmp_path / "frame_baseline" / "v1"
    assert first_kwargs["eval_split"] == "val"
    assert first_kwargs["resolutions"] == (1280, 640)
    assert first_kwargs["eval_pytorch"] is False
    assert first_kwargs["max_calibration_images"] == 500
    assert first_kwargs["upload_to_hf"] is False
    assert first_kwargs["source_path"] == cli.MANIFESTS_DIR / "sources" / "hf.json"

    custom_root = tmp_path / "custom"
    assert cli.main([
        "--split", "test",
        "--resolutions", "640",
        "--dataset-root", str(custom_root),
        "--no-source-check",
        "--calibration-images", "128",
        "--calibrate-method", "Entropy",
        "--n-warmup", "2",
        "--n-benchmark", "5",
        "--eval-pytorch",
        "--upload",
        "--hf-tag", "cpu-v1",
    ]) == 0

    second_args, second_kwargs = mock_run.call_args
    assert second_args[0] == custom_root
    assert second_kwargs["eval_split"] == "test"
    assert second_kwargs["resolutions"] == (640,)
    assert second_kwargs["source_path"] is None
    assert second_kwargs["max_calibration_images"] == 128
    assert second_kwargs["calibrate_method"] == "Entropy"
    assert second_kwargs["n_warmup"] == 2
    assert second_kwargs["n_benchmark"] == 5
    assert second_kwargs["eval_pytorch"] is True
    assert second_kwargs["upload_to_hf"] is True
    assert second_kwargs["hf_tag"] == "cpu-v1"


@pytest.mark.parametrize(
    "bad_argv",
    [
        ["--resolutions", "0"],
        ["--resolutions", "650"],
        ["--hf-tag", "cpu-v1"],
        ["--calibration-images", "0"],
    ],
)
def test_quantize_cpu_cli_rejects_invalid_arguments(bad_argv):
    cli = _load_quantize_cli_module()
    with pytest.raises(SystemExit):
        cli.main(bad_argv)


def test_load_quantization_reports_and_plots_create_pngs(tmp_path):
    from src.utils.visualization import (
        load_quantization_reports,
        plot_quantization_ablation_dashboard,
        plot_quantization_curves,
    )

    quant_dir = tmp_path / "quantization"
    with pytest.raises(ValueError, match="eval_split"):
        load_quantization_reports(quant_dir, eval_split="train")
    with pytest.raises(FileNotFoundError):
        load_quantization_reports(quant_dir, eval_split="val")

    rows = []
    stage_details = {}
    for imgsz, base_ms, base_ap in ((1280, 100.0, 0.90), (640, 30.0, 0.80)):
        rows.append({
            "imgsz": imgsz,
            "stage": "pytorch_fp32",
            "size_mb": 5.5,
            "compute_mean_ms": base_ms,
            "e2e_mean_ms": base_ms + 10.0,
            "e2e_fps": 1000.0 / (base_ms + 10.0),
            "mAP50": float("nan"),
            "AP50": float("nan"),
            "AP50_small": float("nan"),
            "Neg FPR @ R>=0.85 (%)": float("nan"),
            "Recall_small @ Neg FPR<=5%": float("nan"),
            "tp_conf_p10": float("nan"),
            "tp_conf_median": float("nan"),
            "tp_conf_p90": float("nan"),
        })
        for stage, factor in (("onnx_fp32", 0.8), ("onnx_int8", 0.4), ("openvino_fp32", 0.7), ("openvino_int8", 0.3)):
            e2e_ms = base_ms * factor
            rows.append({
                "imgsz": imgsz,
                "stage": stage,
                "size_mb": 3.0 if "int8" in stage else 10.0,
                "compute_mean_ms": e2e_ms * 0.8,
                "e2e_mean_ms": e2e_ms,
                "e2e_fps": 1000.0 / e2e_ms,
                "mAP50": base_ap,
                "AP50": base_ap,
                "AP50_small": base_ap - 0.15,
                "Neg FPR @ R>=0.85 (%)": 6.0,
                "Recall_small @ Neg FPR<=5%": 0.65,
                "tp_conf_p10": 0.4,
                "tp_conf_median": 0.8,
                "tp_recall_ge_0_30": 0.95,
                "tp_conf_p90": 0.95,
            })
            stage_details[f"{stage}_{imgsz}"] = {
                "by_scale": [{"Scale": "small", "GT": 10, "AP50": base_ap - 0.15}],
                "curves": {
                    "thresholds": [0.1, 0.5, 0.9],
                    "precision": [0.7, 0.9, 0.98],
                    "recall": [0.95, 0.85, 0.6],
                    "recall_small": [0.8, 0.65, 0.4],
                    "neg_fa_rate": [12.0, 4.0, 0.5],
                },
            }

    report_dir = quant_dir / "reports" / "val"
    report_dir.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(report_dir / "ablation_results.csv", index=False)
    meta = {
        "eval_split": "val",
        "summary_rows": rows,
        "stage_details": stage_details,
    }
    (report_dir / "quantization_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    loaded_df, loaded_meta = load_quantization_reports(quant_dir, eval_split="val")
    assert len(loaded_df) == 10

    dash_png = report_dir / "dashboard.png"
    curves_png = report_dir / "curves.png"
    plot_quantization_ablation_dashboard(loaded_df, loaded_meta, save_path=dash_png, show=False)
    plot_quantization_curves(loaded_meta, save_path=curves_png, show=False)

    assert dash_png.is_file() and dash_png.stat().st_size > 0
    assert curves_png.is_file() and curves_png.stat().st_size > 0


def test_quantize_onnx_fallback_excludes_full_head_on_initial_failure(tmp_path, monkeypatch):
    calls = []
    def fake_quantize(src, dst, cal, *, imgsz, calibrate_method="MinMax", exclude_head_convs=False):
        calls.append(exclude_head_convs)
        if not exclude_head_convs:
            raise RuntimeError("partial head qdq failed")
        dst.write_bytes(b"qdq-fallback")
        return dst

    fp32 = _make_conv_sigmoid_onnx(tmp_path / "fp32.onnx")
    monkeypatch.setattr(quantization, "quantize_onnx_int8", fake_quantize)

    out, excluded, used_fallback = quantization._quantize_onnx_with_fallback(
        fp32,
        tmp_path / "int8.onnx",
        [tmp_path / "cal.jpg"],
        imgsz=32,
        calibrate_method="MinMax",
    )

    assert calls == [False, True]
    assert used_fallback is True
    assert out.is_file()
    assert "/model.23/Add" in excluded


def test_ablation_uploads_each_int8_immediately_and_resumes_from_failed_stage(tmp_path, monkeypatch):
    rows = [
        {"filename": "quad_rgb_sky_001_f0001.jpg", "split": "train", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_002_f0001.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_002_f0002.jpg", "split": "val", "category": "negative", "excluded_at": ""},
    ]
    root, manifest_path = _write_calibration_bundle(tmp_path, rows)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    quant_dir = tmp_path / "quant"
    working_dir = tmp_path / "work"

    fail_openvino_int8 = True
    export_history = []
    uploaded_models = []

    def fake_export(stage_name, pt, qdir, cal_imgs, cal_yaml, *, imgsz, model_stem, calibrate_method, skip_existing=False):
        export_history.append((stage_name, skip_existing))
        res_dir = Path(qdir) / str(imgsz)
        res_dir.mkdir(parents=True, exist_ok=True)
        if stage_name == "openvino_int8" and fail_openvino_int8:
            raise RuntimeError("openvino int8 exploded")
        if stage_name in {"onnx_fp32", "onnx_int8"}:
            out = _make_conv_sigmoid_onnx(res_dir / f"{model_stem}_{stage_name}_{imgsz}.onnx")
            return out, ({"excluded_nodes": [], "onnx_fallback_full_head": False} if stage_name == "onnx_int8" else {})
        ov_dir = res_dir / f"{model_stem}_{stage_name}_{imgsz}_openvino_model"
        ov_dir.mkdir(exist_ok=True)
        (ov_dir / "model.xml").write_text("<net/>", encoding="utf-8")
        (ov_dir / "model.bin").write_bytes(b"bin")
        return ov_dir, {}

    def fake_single_upload(artifact_path, imgsz, stage_name, **kwargs):
        uploaded_models.append((imgsz, stage_name, Path(artifact_path).name))

    monkeypatch.setattr(quantization, "_export_single_stage", fake_export)
    monkeypatch.setattr(
        quantization,
        "benchmark_stage_model",
        lambda *a, **k: {
            "compute_mean_ms": 10.0,
            "compute_std_ms": 0.5,
            "compute_p50_ms": 10.0,
            "compute_p95_ms": 11.0,
            "compute_p99_ms": 12.0,
            "compute_fps": 100.0,
            "compute_fps_p50": 100.0,
            "e2e_mean_ms": 12.0,
            "e2e_std_ms": 0.5,
            "e2e_p50_ms": 12.0,
            "e2e_p95_ms": 13.0,
            "e2e_p99_ms": 14.0,
            "e2e_fps": 83.3,
            "e2e_fps_p50": 83.3,
            "rss_mb": 100.0,
            "peak_rss_mb": 110.0,
        },
    )
    monkeypatch.setattr(
        quantization,
        "evaluate_quantized_stage",
        lambda *a, **k: {
            "metrics": {**quantization._empty_eval_metrics(), "mAP50": 0.9, "AP50": 0.89, "AP50_small": 0.7},
            "by_scale": [],
            "curves": {},
        },
    )
    mock_reports_upload = Mock(side_effect=RuntimeError("hf service down"))
    monkeypatch.setattr(quantization, "verify_hf_access", Mock(return_value=("u/r", "t")))
    monkeypatch.setattr(quantization, "upload_single_quantized_artifact_to_hf", fake_single_upload)
    monkeypatch.setattr(quantization, "upload_quantization_reports_to_hf", mock_reports_upload)

    first = quantization.run_cpu_quantization_ablation(
        root,
        weights_path=weights,
        quantization_dir=quant_dir,
        working_dir=working_dir,
        resolutions=(640,),
        stages=("onnx_fp32", "onnx_int8", "openvino_int8"),
        n_warmup=1,
        n_benchmark=1,
        manifest_path=manifest_path,
        upload_to_hf=True,
    )

    assert len(first["failed_stages"]) == 1
    assert first["failed_stages"][0]["stage"] == "openvino_int8"
    assert uploaded_models == [(640, "onnx_int8", "yolo26n_run_with_negatives_best_onnx_int8_640.onnx")]
    df1 = first["results_df"].set_index("stage")
    assert df1.loc["onnx_fp32", "status"] == "ok"
    assert df1.loc["onnx_int8", "status"] == "ok"
    assert df1.loc["onnx_int8", "hf_uploaded"] == True
    assert df1.loc["openvino_int8", "status"] == "failed"
    assert first["csv_path"].is_file()
    mock_reports_upload.assert_called_once()

    fail_openvino_int8 = False
    export_history.clear()
    uploaded_models.clear()
    mock_reports_upload.reset_mock()
    mock_reports_upload.side_effect = None

    resumed = quantization.run_cpu_quantization_ablation(
        root,
        weights_path=weights,
        quantization_dir=quant_dir,
        working_dir=working_dir,
        resolutions=(640,),
        stages=("onnx_fp32", "onnx_int8", "openvino_int8"),
        resume=True,
        n_warmup=1,
        n_benchmark=1,
        manifest_path=manifest_path,
        upload_to_hf=True,
    )

    assert resumed["failed_stages"] == []
    assert export_history == [("openvino_int8", True)]
    assert uploaded_models == [(640, "openvino_int8", "yolo26n_run_with_negatives_best_openvino_int8_640_openvino_model")]
    df2 = resumed["results_df"].set_index("stage")
    assert all(df2["status"] == "ok")
    mock_reports_upload.assert_called_once()


def test_quantize_cpu_cli_writes_log_file_and_returns_nonzero_on_failure(tmp_path, monkeypatch):
    cli = _load_quantize_cli_module()
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))

    import pandas as pd

    monkeypatch.setattr(
        cli,
        "run_cpu_quantization_ablation",
        Mock(return_value={
            "results_df": pd.DataFrame([{"imgsz": 1280, "stage": "onnx_int8", "status": "failed"}]),
            "csv_path": tmp_path / "working" / "quantization" / "reports" / "val" / "ablation_results.csv",
            "metadata_path": tmp_path / "working" / "quantization" / "reports" / "val" / "quantization_metadata.json",
            "failed_stages": [{"imgsz": 1280, "stage": "onnx_int8", "error": "boom"}],
        }),
    )

    code = cli.main(["--resume", "--skip-existing-exports", "--stages", "onnx_int8"])
    log_file = tmp_path / "working" / "quantization" / "reports" / "val" / "quantize_cpu.log"
    assert code == 1
    assert log_file.is_file()




def test_try_hf_operation_disables_xet_times_out_and_reraises_permission_error(monkeypatch):
    import threading
    from huggingface_hub import constants as hf_constants
    from huggingface_hub.utils._runtime import is_xet_available

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_XET", False)

    stop_event = threading.Event()

    def slow_upload():
        stop_event.wait(timeout=2.0)

    try:
        assert quantization._try_hf_operation("slow upload", slow_upload, timeout_seconds=0.05) is False
        assert hf_constants.HF_HUB_DISABLE_XET is True
        assert is_xet_available() is False
    finally:
        stop_event.set()

    with pytest.raises(PermissionError, match="read-only"):
        quantization._try_hf_operation(
            "auth check",
            Mock(side_effect=PermissionError("read-only")),
            timeout_seconds=1.0,
        )


def test_ablation_checkpoints_stage_before_hf_upload_timeout(tmp_path, monkeypatch):
    import threading

    rows = [
        {"filename": "quad_rgb_sky_001_f0001.jpg", "split": "train", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_002_f0001.jpg", "split": "val", "category": "positive", "excluded_at": ""},
    ]
    root, manifest_path = _write_calibration_bundle(tmp_path, rows)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    quant_dir = tmp_path / "quant"
    working_dir = tmp_path / "work"
    events = []
    csv_existed_before_upload = []
    stop_event = threading.Event()

    def fake_export(stage_name, pt, qdir, cal_imgs, cal_yaml, *, imgsz, model_stem, calibrate_method, skip_existing=False):
        events.append("export")
        res_dir = Path(qdir) / str(imgsz)
        res_dir.mkdir(parents=True, exist_ok=True)
        out = _make_conv_sigmoid_onnx(res_dir / f"{model_stem}_{stage_name}_{imgsz}.onnx")
        return out, {"excluded_nodes": [], "onnx_fallback_full_head": False}

    def fake_bench(*args, **kwargs):
        events.append("benchmark")
        return {
            "compute_mean_ms": 10.0,
            "compute_std_ms": 0.5,
            "compute_p50_ms": 10.0,
            "compute_p95_ms": 11.0,
            "compute_p99_ms": 12.0,
            "compute_fps": 100.0,
            "compute_fps_p50": 100.0,
            "e2e_mean_ms": 12.0,
            "e2e_std_ms": 0.5,
            "e2e_p50_ms": 12.0,
            "e2e_p95_ms": 13.0,
            "e2e_p99_ms": 14.0,
            "e2e_fps": 83.3,
            "e2e_fps_p50": 83.3,
            "rss_mb": 100.0,
            "peak_rss_mb": 110.0,
        }

    def fake_eval(*args, **kwargs):
        events.append("evaluate")
        return {
            "metrics": {**quantization._empty_eval_metrics(), "mAP50": 0.91, "AP50": 0.90, "AP50_small": 0.72},
            "by_scale": [],
            "curves": {},
        }

    def hanging_upload(artifact_path, imgsz, stage_name, **kwargs):
        events.append("upload")
        csv_existed_before_upload.append((quant_dir / "reports" / "val" / "ablation_results.csv").is_file())
        stop_event.wait(timeout=2.0)

    monkeypatch.setattr(quantization, "HF_UPLOAD_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(quantization, "_export_single_stage", fake_export)
    monkeypatch.setattr(quantization, "benchmark_stage_model", fake_bench)
    monkeypatch.setattr(quantization, "evaluate_quantized_stage", fake_eval)
    monkeypatch.setattr(quantization, "verify_hf_access", Mock(return_value=("u/r", "t")))
    monkeypatch.setattr(quantization, "upload_single_quantized_artifact_to_hf", hanging_upload)
    monkeypatch.setattr(quantization, "upload_quantization_reports_to_hf", Mock())

    try:
        result = quantization.run_cpu_quantization_ablation(
            root,
            weights_path=weights,
            quantization_dir=quant_dir,
            working_dir=working_dir,
            resolutions=(640,),
            stages=("onnx_int8",),
            n_warmup=1,
            n_benchmark=1,
            manifest_path=manifest_path,
            upload_to_hf=True,
        )
    finally:
        stop_event.set()

    assert events == ["export", "benchmark", "evaluate", "upload"]
    assert csv_existed_before_upload == [True]
    assert result["failed_stages"] == []
    df = result["results_df"]
    assert df.loc[0, "status"] == "ok"
    assert bool(df.loc[0, "hf_uploaded"]) is False
    assert df.loc[0, "AP50"] == pytest.approx(0.90)




def test_get_process_memory_mb_returns_positive_on_current_process():
    rss_mb, peak_rss_mb = quantization._get_process_memory_mb()
    assert rss_mb > 0.0
    assert peak_rss_mb >= rss_mb


def test_quantize_onnx_int8_passes_moving_average_and_percentile_options(tmp_path, monkeypatch):
    fp32 = _make_conv_sigmoid_onnx(tmp_path / "model.onnx", ir_version=9)
    cal_img = tmp_path / "cal.jpg"
    assert cv2.imwrite(str(cal_img), np.full((32, 32, 3), 120, dtype=np.uint8))

    captured = []

    def fake_quantize_static(**kwargs):
        captured.append(kwargs)
        shutil_dst = Path(kwargs["model_output"])
        shutil_dst.write_bytes(fp32.read_bytes())

    import onnxruntime as ort
    import onnxruntime.quantization as ort_q

    monkeypatch.setattr(quantization, "_preprocess_onnx_for_quantization", lambda src, dst: src)
    monkeypatch.setattr(ort_q, "quantize_static", fake_quantize_static)
    monkeypatch.setattr(ort, "InferenceSession", Mock())

    quantization.quantize_onnx_int8(
        fp32,
        tmp_path / "mavg.onnx",
        [cal_img],
        imgsz=32,
        calibrate_method="MinMax",
        moving_average=True,
    )
    assert captured[-1]["extra_options"]["CalibMovingAverage"] is True
    assert captured[-1]["extra_options"]["CalibMovingAverageConstant"] == pytest.approx(0.01)

    quantization.quantize_onnx_int8(
        fp32,
        tmp_path / "pct.onnx",
        [cal_img],
        imgsz=32,
        calibrate_method="Percentile",
        calibration_symmetric=False,
        percentile=99.99,
        exclude_head_convs=True,
    )
    assert captured[-1]["extra_options"]["CalibTensorRangeSymmetric"] is False
    assert captured[-1]["extra_options"]["CalibPercentile"] == pytest.approx(99.99)

    with pytest.raises(ValueError, match="MinMax"):
        quantization.quantize_onnx_int8(
            fp32,
            tmp_path / "bad.onnx",
            [cal_img],
            imgsz=32,
            calibrate_method="Entropy",
            moving_average=True,
        )


def test_quantize_onnx_nncf_and_export_openvino_nncf_preserve_metadata(tmp_path, monkeypatch):
    import nncf
    import openvino as ov
    import onnxruntime as ort

    fp32_onnx = _make_conv_sigmoid_onnx(tmp_path / "model.onnx", ir_version=9)
    cal_img = tmp_path / "cal.jpg"
    assert cv2.imwrite(str(cal_img), np.full((32, 32, 3), 120, dtype=np.uint8))

    nncf_calls = []

    def fake_nncf_quantize(**kwargs):
        nncf_calls.append(kwargs)
        return kwargs["model"]

    monkeypatch.setattr(nncf, "quantize", fake_nncf_quantize)
    monkeypatch.setattr(ort, "InferenceSession", Mock())

    out_onnx = quantization.quantize_onnx_int8_nncf(
        fp32_onnx,
        tmp_path / "nncf.onnx",
        [cal_img],
        imgsz=32,
        exclude_head_convs=True,
        fast_bias_correction=True,
    )
    assert out_onnx.is_file()
    assert nncf_calls[-1]["preset"] == nncf.QuantizationPreset.MIXED
    assert nncf_calls[-1]["fast_bias_correction"] is True
    assert "/model.23/Add" in nncf_calls[-1]["ignored_scope"].names
    assert {p.key: p.value for p in onnx.load(str(out_onnx)).metadata_props}["task"] == "detect"

    ov_fp32_dir = tmp_path / "fp32_openvino_model"
    ov_fp32_dir.mkdir()
    (ov_fp32_dir / "best.xml").write_text("<net/>", encoding="utf-8")
    (ov_fp32_dir / "best.bin").write_bytes(b"bin")
    (ov_fp32_dir / "metadata.yaml").write_text("task: detect\n", encoding="utf-8")

    fake_core = SimpleNamespace(read_model=Mock(return_value="ov_model_obj"))
    monkeypatch.setattr(ov, "Core", Mock(return_value=fake_core))

    def fake_save_model(model_obj, xml_path, compress_to_fp16=False):
        p = Path(xml_path)
        p.write_text("<net/>", encoding="utf-8")
        p.with_suffix(".bin").write_bytes(b"bin")

    monkeypatch.setattr(ov, "save_model", fake_save_model)

    ov_out = quantization.export_openvino_int8_nncf(
        ov_fp32_dir,
        tmp_path / "ov_nncf_openvino_model",
        [cal_img],
        imgsz=32,
        fast_bias_correction=False,
    )
    assert (ov_out / "best.xml").is_file()
    assert (ov_out / "metadata.yaml").is_file()
    assert nncf_calls[-1]["fast_bias_correction"] is False
    assert r".*model\.23/.*" in nncf_calls[-1]["ignored_scope"].patterns


def test_run_ptq_variant_ablation_seeds_baseline_skips_hf_and_filters_resolutions(tmp_path, monkeypatch):
    from src.utils.visualization import load_quantization_reports

    rows = [
        {"filename": "quad_rgb_sky_001_f0001.jpg", "split": "train", "category": "positive", "excluded_at": ""},
        {"filename": "quad_rgb_sky_002_f0001.jpg", "split": "val", "category": "positive", "excluded_at": ""},
    ]
    root, manifest_path = _write_calibration_bundle(tmp_path, rows)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    quant_dir = tmp_path / "quant"
    working_dir = tmp_path / "work"
    (quant_dir / "1280").mkdir(parents=True, exist_ok=True)

    base_onnx = _make_conv_sigmoid_onnx(quant_dir / "1280" / "yolo26n_run_with_negatives_best_fp32_1280.onnx")
    base_int8 = _make_conv_sigmoid_onnx(quant_dir / "1280" / "yolo26n_run_with_negatives_best_int8_1280.onnx")
    base_ov_fp32 = quant_dir / "1280" / "yolo26n_run_with_negatives_best_fp32_1280_openvino_model"
    base_ov_fp32.mkdir(parents=True)
    (base_ov_fp32 / "model.xml").write_text("<net/>", encoding="utf-8")
    (base_ov_fp32 / "model.bin").write_bytes(b"bin")
    base_ov_int8 = quant_dir / "1280" / "yolo26n_run_with_negatives_best_int8_1280_openvino_model"
    base_ov_int8.mkdir(parents=True)
    (base_ov_int8 / "model.xml").write_text("<net/>", encoding="utf-8")
    (base_ov_int8 / "model.bin").write_bytes(b"bin")

    report_val_dir = quant_dir / "reports" / "val"
    report_val_dir.mkdir(parents=True)
    seed_sha = quantization._file_sha256(weights)
    seed_rows = [
        {
            "imgsz": 1280,
            "stage": "onnx_fp32",
            "status": "ok",
            "artifact": str(base_onnx),
            "size_mb": 9.8,
            "compute_mean_ms": 300.0,
            "e2e_mean_ms": 380.0,
            "e2e_fps": 2.6,
            "mAP50": 0.96,
            "mAP50-95": 0.82,
            "AP50": 0.95,
            "AP50_small": 0.91,
            "Recall_small @ Neg FPR<=5%": 0.70,
            "Neg FPR @ R>=0.85 (%)": 4.0,
        },
        {
            "imgsz": 1280,
            "stage": "onnx_int8",
            "status": "ok",
            "artifact": str(base_int8),
            "size_mb": 3.5,
            "compute_mean_ms": 350.0,
            "e2e_mean_ms": 580.0,
            "e2e_fps": 1.7,
            "mAP50": 0.90,
            "mAP50-95": 0.72,
            "AP50": 0.88,
            "AP50_small": 0.87,
            "Recall_small @ Neg FPR<=5%": 0.80,
            "Neg FPR @ R>=0.85 (%)": 16.0,
        },
        {
            "imgsz": 1280,
            "stage": "openvino_fp32",
            "status": "ok",
            "artifact": str(base_ov_fp32),
            "size_mb": 9.8,
            "compute_mean_ms": 270.0,
            "e2e_mean_ms": 540.0,
            "e2e_fps": 1.85,
            "mAP50": 0.96,
            "mAP50-95": 0.82,
            "AP50": 0.95,
            "AP50_small": 0.92,
            "Recall_small @ Neg FPR<=5%": 0.66,
            "Neg FPR @ R>=0.85 (%)": 5.4,
        },
        {
            "imgsz": 1280,
            "stage": "openvino_int8",
            "status": "ok",
            "artifact": str(base_ov_int8),
            "size_mb": 3.5,
            "compute_mean_ms": 138.0,
            "e2e_mean_ms": 600.0,
            "e2e_fps": 1.66,
            "mAP50": 0.955,
            "mAP50-95": 0.795,
            "AP50": 0.95,
            "AP50_small": 0.94,
            "Recall_small @ Neg FPR<=5%": 0.75,
            "Neg FPR @ R>=0.85 (%)": 6.8,
        },
    ]
    (report_val_dir / "quantization_metadata.json").write_text(
        json.dumps({"source_weights_sha256": seed_sha, "summary_rows": seed_rows, "stage_details": {}}),
        encoding="utf-8",
    )

    exported = []

    def fake_export_ptq(variant_name, spec, *, source_pt, quantization_dir, calibration_images, imgsz, model_stem, skip_existing):
        exported.append((imgsz, variant_name))
        out_dir = Path(quantization_dir) / "ptq_ablation" / str(imgsz)
        out_dir.mkdir(parents=True, exist_ok=True)
        if spec["engine"] == "nncf_openvino":
            ov_dir = out_dir / f"{model_stem}_{variant_name}_{imgsz}_openvino_model"
            ov_dir.mkdir(exist_ok=True)
            (ov_dir / "model.xml").write_text("<net/>", encoding="utf-8")
            (ov_dir / "model.bin").write_bytes(b"bin")
            return ov_dir
        return _make_conv_sigmoid_onnx(out_dir / f"{model_stem}_{variant_name}_{imgsz}.onnx")

    def fake_ensure_fp32(fp32_stage, source_pt, qdir, *, imgsz, model_stem):
        exported.append((imgsz, fp32_stage))
        res_dir = Path(qdir) / str(imgsz)
        res_dir.mkdir(parents=True, exist_ok=True)
        return _make_conv_sigmoid_onnx(res_dir / f"{model_stem}_fp32_{imgsz}.onnx")

    monkeypatch.setattr(quantization, "verify_hf_access", Mock(side_effect=AssertionError("HF must not be called")))
    monkeypatch.setattr(quantization, "_export_ptq_variant", fake_export_ptq)
    monkeypatch.setattr(quantization, "_ensure_fp32_for_ptq", fake_ensure_fp32)
    monkeypatch.setattr(
        quantization,
        "benchmark_stage_model",
        lambda *a, **k: {
            "compute_mean_ms": 50.0,
            "compute_std_ms": 1.0,
            "compute_p50_ms": 50.0,
            "compute_p95_ms": 52.0,
            "compute_p99_ms": 53.0,
            "compute_fps": 20.0,
            "compute_fps_p50": 20.0,
            "e2e_mean_ms": 80.0,
            "e2e_std_ms": 1.0,
            "e2e_p50_ms": 80.0,
            "e2e_p95_ms": 82.0,
            "e2e_p99_ms": 83.0,
            "e2e_fps": 12.5,
            "e2e_fps_p50": 12.5,
            "rss_mb": 120.0,
            "peak_rss_mb": 130.0,
        },
    )
    monkeypatch.setattr(
        quantization,
        "evaluate_quantized_stage",
        lambda *a, **k: {
            "metrics": {**quantization._empty_eval_metrics(), "mAP50": 0.94, "AP50": 0.93, "AP50_small": 0.90, "Neg FPR @ R>=0.85 (%)": 5.0, "Recall_small @ Neg FPR<=5%": 0.72},
            "by_scale": [],
            "curves": {},
        },
    )

    result = quantization.run_ptq_variant_ablation(
        root,
        weights_path=weights,
        quantization_dir=quant_dir,
        working_dir=working_dir,
        resolutions=(1280, 640),
        variants=(
            "onnx_int8_minmax_head_int8",
            "onnx_int8_entropy_head_fp32",
            "openvino_int8_nncf_full_bias_head_fp32",
        ),
        eval_split="val",
        resume=True,
        n_warmup=1,
        n_benchmark=1,
        manifest_path=manifest_path,
    )

    assert (1280, "onnx_int8_minmax_head_int8") not in exported
    assert (640, "onnx_fp32") in exported
    assert (640, "onnx_int8_minmax_head_int8") in exported
    assert (1280, "onnx_int8_entropy_head_fp32") in exported
    assert (640, "onnx_int8_entropy_head_fp32") in exported
    assert (1280, "openvino_int8_nncf_full_bias_head_fp32") in exported
    assert (640, "openvino_int8_nncf_full_bias_head_fp32") not in exported
    assert result["csv_path"] == quant_dir / "reports" / "ptq_ablation_val" / "ablation_results.csv"

    loaded_df, loaded_meta = load_quantization_reports(quant_dir, eval_split="ptq_ablation_val")
    assert loaded_meta["report_split"] == "ptq_ablation_val"
    df_idx = loaded_df.set_index(["imgsz", "stage"])
    assert df_idx.loc[(1280, "onnx_int8_entropy_head_fp32"), "ap50_delta_vs_fp32"] == pytest.approx(-0.02)
    assert df_idx.loc[(1280, "openvino_int8_nncf_full_bias_head_fp32"), "ap50_delta_vs_fp32"] == pytest.approx(-0.02)


def test_quantize_cpu_cli_ptq_ablation_rejects_upload_and_invokes_runner(tmp_path, monkeypatch):
    cli = _load_quantize_cli_module()
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))

    with pytest.raises(SystemExit):
        cli.main(["--ptq-ablation", "--upload"])

    import pandas as pd

    mock_ptq = Mock(return_value={
        "results_df": pd.DataFrame([{"imgsz": 1280, "stage": "onnx_int8_entropy_head_fp32", "status": "ok"}]),
        "csv_path": tmp_path / "working" / "quantization" / "reports" / "ptq_ablation_val" / "ablation_results.csv",
        "metadata_path": tmp_path / "working" / "quantization" / "reports" / "ptq_ablation_val" / "quantization_metadata.json",
        "failed_stages": [],
    })
    monkeypatch.setattr(cli, "run_ptq_variant_ablation", mock_ptq)

    code = cli.main(["--ptq-ablation", "--resume", "--variants", "onnx_int8_entropy_head_fp32"])
    log_file = tmp_path / "working" / "quantization" / "reports" / "ptq_ablation_val" / "quantize_cpu.log"
    assert code == 0
    assert log_file.is_file()
    assert mock_ptq.call_args.kwargs["variants"] == ("onnx_int8_entropy_head_fp32",)


def test_plot_ptq_variant_ablation_dashboard_handles_failed_rows(tmp_path):
    import matplotlib
    import pandas as pd
    from src.utils.visualization import plot_ptq_variant_ablation_dashboard

    matplotlib.use("Agg")
    df = pd.DataFrame([
        {
            "imgsz": 1280,
            "stage": "onnx_fp32",
            "status": "ok",
            "compute_fps": 3.2,
            "AP50": 0.95,
            "AP50_small": 0.91,
            "Neg FPR @ R>=0.85 (%)": 4.0,
            "tp_conf_p10": 0.92,
            "tp_conf_median": 0.97,
        },
        {
            "imgsz": 1280,
            "stage": "onnx_int8_moving_avg_head_fp32",
            "status": "ok",
            "compute_fps": 3.1,
            "AP50": 0.936,
            "AP50_small": 0.937,
            "Neg FPR @ R>=0.85 (%)": 5.4,
            "tp_conf_p10": 0.85,
            "tp_conf_median": 0.97,
        },
        {
            "imgsz": 1280,
            "stage": "onnx_int8_entropy_head_fp32",
            "status": "failed",
            "compute_fps": np.nan,
            "AP50": np.nan,
            "AP50_small": np.nan,
            "Neg FPR @ R>=0.85 (%)": np.nan,
            "tp_conf_p10": np.nan,
            "tp_conf_median": np.nan,
        },
        {
            "imgsz": 640,
            "stage": "openvino_int8_ultralytics",
            "status": "ok",
            "compute_fps": 38.6,
            "AP50": 0.782,
            "AP50_small": 0.428,
            "Neg FPR @ R>=0.85 (%)": np.nan,
            "tp_conf_p10": 0.28,
            "tp_conf_median": 0.95,
        },
    ])
    out_png = tmp_path / "ptq_dash.png"
    plot_ptq_variant_ablation_dashboard(df, save_path=out_png, show=False)
    assert out_png.is_file()


def test_rebalance_small_quotas_boosts_small_without_reducing_negatives():
    stratum_sizes = {
        ("negative", "quad", "rgb", "sky", "none"): 100,
        ("positive", "quad", "rgb", "sky", "large"): 200,
        ("positive", "quad", "rgb", "sky", "medium"): 80,
        ("positive", "quad", "rgb", "sky", "small"): 20,
    }
    base = quantization._allocate_stratum_quotas(stratum_sizes, 100)
    rebalanced = quantization._rebalance_small_quotas(base, stratum_sizes)

    assert sum(rebalanced.values()) == 100
    assert (
        rebalanced[("negative", "quad", "rgb", "sky", "none")]
        == base[("negative", "quad", "rgb", "sky", "none")]
    )
    assert rebalanced[("positive", "quad", "rgb", "sky", "small")] >= 15


def test_prepare_calibration_dataset_stratifies_by_bbox_scale_and_shuffles_order(tmp_path):
    root = tmp_path / "curated"
    manifest_rows = []
    specs = [
        ("negative", "quad_rgb_sky_001", 40, None),
        ("positive", "quad_rgb_sky_002", 45, "0 0.5 0.5 0.4 0.4\n"),
        ("positive", "wing_rgb_city_003", 45, "0 0.5 0.5 0.1 0.1\n"),
        ("positive", "quad_thermal_forest_004", 6, "0 0.5 0.5 0.02 0.04\n"),
    ]
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for category, prefix, count, label_line in specs:
        folder = root / "train" / CATEGORY_DIRS[category]
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(1, count + 1):
            filename = f"{prefix}_f{idx:04d}.jpg"
            img_path = folder / filename
            assert cv2.imwrite(str(img_path), black_frame)
            if label_line is not None:
                img_path.with_suffix(".txt").write_text(label_line, encoding="utf-8")
            manifest_rows.append({
                "filename": filename,
                "split": "train",
                "category": category,
                "excluded_at": "",
            })

    val_folder = root / "val" / CATEGORY_DIRS["positive"]
    val_folder.mkdir(parents=True, exist_ok=True)
    val_path = val_folder / "quad_rgb_sky_900_f0001.jpg"
    assert cv2.imwrite(str(val_path), black_frame)
    val_path.with_suffix(".txt").write_text("0 0.5 0.5 0.02 0.04\n", encoding="utf-8")
    manifest_rows.append({
        "filename": val_path.name,
        "split": "val",
        "category": "positive",
        "excluded_at": "",
    })

    manifest_path = tmp_path / "main.csv"
    write_manifest(manifest_rows, manifest_path)
    working_dir = tmp_path / "working"

    _, cal_42_a = quantization.prepare_calibration_dataset(
        root,
        working_dir,
        manifest_path=manifest_path,
        max_images=24,
        seed=42,
        run_name="cal_a",
    )
    scale_counts = quantization._summarize_calibration_scales(cal_42_a)
    assert len(cal_42_a) == 24
    assert scale_counts["small"] >= round(24 * quantization.DEFAULT_MIN_SMALL_CALIBRATION_FRACTION)
    assert scale_counts["medium"] >= 1
    assert scale_counts["large"] >= 1
    assert scale_counts["negative"] >= 1
    assert all(path.name != val_path.name for path in cal_42_a)

    names_42_a = [path.name for path in cal_42_a]
    assert names_42_a != sorted(names_42_a)

    _, cal_42_b = quantization.prepare_calibration_dataset(
        root,
        working_dir,
        manifest_path=manifest_path,
        max_images=24,
        seed=42,
        run_name="cal_b",
    )
    assert [path.name for path in cal_42_b] == names_42_a

    _, cal_99 = quantization.prepare_calibration_dataset(
        root,
        working_dir,
        manifest_path=manifest_path,
        max_images=24,
        seed=99,
        run_name="cal_c",
    )
    assert [path.name for path in cal_99] != names_42_a