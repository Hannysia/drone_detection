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

    def fake_export_stage(pt, qdir, cal_imgs, cal_yaml, *, imgsz, model_stem, calibrate_method):
        res_dir = qdir / str(imgsz)
        res_dir.mkdir(parents=True, exist_ok=True)
        staged_pt = res_dir / f"{model_stem}.pt"
        staged_pt.write_bytes(b"pt")
        onnx_fp32 = _make_conv_sigmoid_onnx(res_dir / f"{model_stem}_fp32_{imgsz}.onnx")
        onnx_int8 = _make_conv_sigmoid_onnx(res_dir / f"{model_stem}_int8_{imgsz}.onnx")
        ov_fp32 = res_dir / f"{model_stem}_fp32_{imgsz}_openvino_model"
        ov_int8 = res_dir / f"{model_stem}_int8_{imgsz}_openvino_model"
        for ov_dir in (ov_fp32, ov_int8):
            ov_dir.mkdir()
            (ov_dir / "model.xml").write_text("<net/>", encoding="utf-8")
            (ov_dir / "model.bin").write_bytes(b"bin")
        return {
            "pytorch_fp32": staged_pt,
            "onnx_fp32": onnx_fp32,
            "onnx_int8": onnx_int8,
            "openvino_fp32": ov_fp32,
            "openvino_int8": ov_int8,
        }

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

    monkeypatch.setattr(quantization, "export_stage_models", fake_export_stage)
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