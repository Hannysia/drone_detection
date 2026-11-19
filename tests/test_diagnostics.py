from unittest.mock import Mock
import cv2
import numpy as np
import pytest
import torch

from src.processing import diagnostics
from src.processing.diagnostics import (
    compare_models_shortcut_learning,
    find_sppf_layer,
    letterbox_image,
    plot_false_positive_inline,
    preprocess_image_tensor,
    run_false_positive_inference,
    run_shortcut_analysis,
)
from src.processing.manifests import CATEGORY_DIRS, write_manifest

ConvLayer = type("Conv", (object,), {})
SPPFLayer = type("SPPF", (object,), {})
DetectLayer = type("Detect", (object,), {})


class DummyPyTorchModel:
    """Mock PyTorch model containing a sequential list of dummy layers."""
    def __init__(self, layers: list):
        self.model = layers


def test_find_sppf_layer_dynamically_found():
    layers = [ConvLayer(), SPPFLayer(), DetectLayer()]
    mock_model = DummyPyTorchModel(layers)

    layer, idx = find_sppf_layer(mock_model)

    assert idx == 1
    assert type(layer).__name__ == "SPPF"


def test_find_sppf_layer_fallback_to_index_9():
    layers = [ConvLayer() for _ in range(12)]
    mock_model = DummyPyTorchModel(layers)

    layer, idx = find_sppf_layer(mock_model)

    assert idx == 9


def test_manual_eigencam_computes_normalized_map_and_removes_hook():
    from src.processing.diagnostics import manual_eigencam

    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    net = ToyNet()
    x = torch.randn(1, 3, 8, 8)
    cam_map = manual_eigencam(net, net.conv, x)

    assert cam_map.shape == (8, 8)
    assert cam_map.min() >= 0.0
    assert cam_map.max() <= 1.0
    assert len(net.conv._forward_hooks) == 0


def test_diagnostics_reuses_evaluation_get_detections_boxes():
    from src.processing import diagnostics, evaluation

    assert diagnostics.get_detections_boxes is evaluation.get_detections_boxes


def test_preprocess_image_tensor_uses_1280_letterbox_and_fails_on_unreadable(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unreadable diagnostic image"):
        preprocess_image_tensor(broken, torch.device("cpu"))

    img_path = tmp_path / "wide.jpg"
    assert cv2.imwrite(str(img_path), np.full((100, 200, 3), 255, dtype=np.uint8))

    orig, letterboxed, tensor = preprocess_image_tensor(img_path, torch.device("cpu"))
    assert orig.shape == (100, 200, 3)
    assert letterboxed.shape == (1280, 1280, 3)
    assert tuple(letterboxed[0, 0]) == (114, 114, 114)
    assert tuple(tensor.shape) == (1, 3, 1280, 1280)

    _, crop_box = letterbox_image(orig, 1280)
    assert crop_box == (320, 960, 0, 1280)


def test_plot_false_positive_inline_uses_unpadded_1280_eigencam(tmp_path, monkeypatch):
    img_path = tmp_path / "neg.jpg"
    assert cv2.imwrite(str(img_path), np.zeros((100, 200, 3), dtype=np.uint8))

    cam_calls = []
    monkeypatch.setattr(
        diagnostics,
        "manual_eigencam",
        lambda model, layer, tensor: (cam_calls.append(tuple(tensor.shape)), np.ones((40, 40), dtype=np.float32))[1],
    )
    overlay_spy = Mock(return_value=np.zeros((100, 200, 3), dtype=np.uint8))
    monkeypatch.setattr(diagnostics, "overlay_heatmap", overlay_spy)
    render_spy = Mock()
    monkeypatch.setattr(diagnostics, "plot_false_positive_results", render_spy)

    fp = {
        "path": img_path,
        "n_dets": 1,
        "max_conf": 0.85,
        "boxes": np.array([[10.0, 12.0, 50.0, 40.0]]),
        "confs": np.array([0.85]),
    }

    plot_false_positive_inline(fp, object(), object(), torch.device("cpu"), show=False)

    assert cam_calls == [(1, 3, 1280, 1280)]
    assert overlay_spy.call_args.kwargs["crop_box"] == (320, 960, 0, 1280)
    assert overlay_spy.call_args.kwargs["input_size"] == (1280, 1280)
    render_spy.assert_called_once()


def test_run_false_positive_inference_enforces_manifest_and_imgsz(tmp_path):
    root = tmp_path / "curated"
    rows = [
        {"filename": "tr_neg.jpg", "split": "train", "category": "negative", "excluded_at": ""},
        {"filename": "va_keep.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "va_excl.jpg", "split": "val", "category": "negative", "excluded_at": "temporal"},
    ]
    for item in rows:
        folder = root / item["split"] / CATEGORY_DIRS[item["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(folder / item["filename"]), np.zeros((32, 32, 3), dtype=np.uint8))

    manifest_path = tmp_path / "main.csv"
    write_manifest(rows, manifest_path)

    fake_boxes = Mock()
    fake_boxes.__len__ = Mock(return_value=1)
    fake_boxes.conf = torch.tensor([0.65])
    fake_boxes.xyxy = torch.tensor([[1.0, 2.0, 10.0, 12.0]])
    fake_model = Mock(return_value=[Mock(boxes=fake_boxes)])

    fps = run_false_positive_inference(
        fake_model,
        root,
        conf_thresh=0.25,
        split="val",
        manifest_path=manifest_path,
    )

    fake_model.assert_called_once()
    assert fake_model.call_args.args[0].endswith("va_keep.jpg")
    assert fake_model.call_args.kwargs["imgsz"] == 1280
    assert len(fps) == 1
    assert fps[0]["path"].name == "va_keep.jpg"


def test_run_shortcut_analysis_saves_plots_and_uses_output_dir_fallback(tmp_path, monkeypatch):
    root = tmp_path / "curated"
    rows = [
        {"filename": "va_keep.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "va_drop.jpg", "split": "val", "category": "positive", "excluded_at": "temporal"},
    ]
    for item in rows:
        folder = root / item["split"] / CATEGORY_DIRS[item["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(folder / item["filename"]), np.zeros((64, 64, 3), dtype=np.uint8))

    manifest_path = tmp_path / "main.csv"
    write_manifest(rows, manifest_path)

    output_base = tmp_path / "runs" / "detect"
    fallback_weights = output_base / "demo_run" / "weights" / "best.pt"
    fallback_weights.parent.mkdir(parents=True)
    fallback_weights.write_bytes(b"pt")

    monkeypatch.setattr(diagnostics.config, "OUTPUT_DIR", output_base)
    monkeypatch.setattr(diagnostics, "REPO_ID", None)
    monkeypatch.setattr(diagnostics, "HF_TOKEN", None)

    fake_model = Mock(return_value=[Mock(boxes=[])])
    fake_pytorch = DummyPyTorchModel([SPPFLayer()])
    monkeypatch.setattr(diagnostics, "load_eval_model", lambda path, device: (fake_model, fake_pytorch))
    monkeypatch.setattr(diagnostics, "manual_eigencam", lambda m, l, t: np.ones((8, 8), dtype=np.float32))

    out_dir = tmp_path / "diag_out"
    run_shortcut_analysis(
        tmp_path / "nonexistent.pt",
        root,
        out_dir,
        "demo_run",
        n_images=5,
        manifest_path=manifest_path,
        show=False,
    )

    assert (out_dir / "demo_run" / "cam_000_va_keep.png").is_file()
    assert not (out_dir / "demo_run" / "cam_001_va_drop.png").exists()
    assert (out_dir / "demo_run" / "detection_summary.png").is_file()
    assert fake_model.call_args.kwargs["imgsz"] == 1280

    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        run_shortcut_analysis(
            tmp_path / "missing.pt",
            root,
            out_dir,
            "missing_run",
            manifest_path=manifest_path,
            show=False,
        )


def test_compare_models_shortcut_learning_generates_1x3_pairs(tmp_path, monkeypatch):
    root = tmp_path / "curated"
    rows = [
        {"filename": "va_pos.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "va_neg.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "va_drop.jpg", "split": "val", "category": "negative", "excluded_at": "temporal"},
    ]
    for item in rows:
        folder = root / item["split"] / CATEGORY_DIRS[item["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(folder / item["filename"]), np.zeros((64, 64, 3), dtype=np.uint8))

    manifest_path = tmp_path / "main.csv"
    write_manifest(rows, manifest_path)

    w_pos = tmp_path / "pos.pt"
    w_neg = tmp_path / "neg.pt"
    w_pos.write_bytes(b"pt")
    w_neg.write_bytes(b"pt")

    fp_box = Mock()
    fp_box.__len__ = Mock(return_value=1)
    fp_box.conf = torch.tensor([0.75])
    fp_box.xyxy = torch.tensor([[5.0, 5.0, 20.0, 20.0]])

    def _model_pos(path, **kwargs):
        if "va_neg" in str(path):
            return [Mock(boxes=fp_box)]
        return [Mock(boxes=[])]

    def _model_neg(path, **kwargs):
        return [Mock(boxes=[])]

    fake_left = Mock(side_effect=_model_pos)
    fake_right = Mock(side_effect=_model_neg)
    fake_pt = DummyPyTorchModel([SPPFLayer()])

    def _fake_load(weights_path, device):
        return (fake_left, fake_pt) if Path_name(weights_path) == "pos.pt" else (fake_right, fake_pt)

    from pathlib import Path
    def Path_name(value):
        return Path(value).name

    monkeypatch.setattr(diagnostics, "load_eval_model", _fake_load)
    monkeypatch.setattr(diagnostics, "manual_eigencam", lambda m, l, t: np.ones((8, 8), dtype=np.float32))

    out_dir = tmp_path / "comparison"
    result = compare_models_shortcut_learning(
        {"Positives Only": w_pos, "With Negatives": w_neg},
        root,
        out_dir,
        n_negative_pairs=5,
        n_positive_pairs=5,
        manifest_path=manifest_path,
        show_negatives=False,
        show_positives=False,
    )

    summary = result["summary"]
    assert list(summary["Run"]) == ["Positives Only", "With Negatives"]
    assert list(summary["Val Negatives"]) == [1, 1]
    assert list(summary["FP Frames"]) == [1, 0]
    assert list(summary["FP Rate (%)"]) == [100.0, 0.0]
    assert (out_dir / "negatives_pairs" / "pair_00_va_neg.png").is_file()
    assert (out_dir / "positives_pairs" / "pair_00_va_pos.png").is_file()

    with pytest.raises(ValueError, match="exactly 2"):
        compare_models_shortcut_learning({"Only One": w_pos}, root, out_dir, manifest_path=manifest_path)


def test_compare_models_shortcut_learning_picks_unique_videos_first(tmp_path, monkeypatch):
    root = tmp_path / "curated"
    rows = [
        {"filename": "vidA_001_f0001.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "vidA_001_f0002.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "vidA_001_f0003.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "vidB_002_f0001.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "vidC_003_f0001.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "posA_010_f0001.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "posA_010_f0002.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "posB_011_f0001.jpg", "split": "val", "category": "positive", "excluded_at": ""},
    ]
    for item in rows:
        folder = root / item["split"] / CATEGORY_DIRS[item["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(folder / item["filename"]), np.zeros((64, 64, 3), dtype=np.uint8))

    manifest_path = tmp_path / "main.csv"
    write_manifest(rows, manifest_path)

    w_left = tmp_path / "left.pt"
    w_right = tmp_path / "right.pt"
    w_left.write_bytes(b"pt")
    w_right.write_bytes(b"pt")

    fp_confs = {
        "vidA_001_f0001.jpg": 0.95,
        "vidA_001_f0002.jpg": 0.90,
        "vidA_001_f0003.jpg": 0.85,
        "vidB_002_f0001.jpg": 0.70,
    }

    def _make_boxes(conf_val):
        box = Mock()
        box.__len__ = Mock(return_value=1)
        box.conf = torch.tensor([conf_val])
        box.xyxy = torch.tensor([[5.0, 5.0, 20.0, 20.0]])
        return box

    def _model_left(path, **kwargs):
        for fname, conf_val in fp_confs.items():
            if str(path).endswith(fname):
                return [Mock(boxes=_make_boxes(conf_val))]
        return [Mock(boxes=[])]

    def _model_right(path, **kwargs):
        return [Mock(boxes=[])]

    fake_pt = DummyPyTorchModel([SPPFLayer()])
    fake_left = Mock(side_effect=_model_left)
    fake_right = Mock(side_effect=_model_right)
    monkeypatch.setattr(
        diagnostics,
        "load_eval_model",
        lambda p, d: (fake_left, fake_pt) if str(p).endswith("left.pt") else (fake_right, fake_pt),
    )
    monkeypatch.setattr(diagnostics, "manual_eigencam", lambda m, l, t: np.ones((8, 8), dtype=np.float32))

    out_dir = tmp_path / "diverse_out"
    compare_models_shortcut_learning(
        {"Left": w_left, "Right": w_right},
        root,
        out_dir,
        n_negative_pairs=3,
        n_positive_pairs=2,
        manifest_path=manifest_path,
        show_negatives=False,
        show_positives=False,
    )

    neg_names = sorted(p.name for p in (out_dir / "negatives_pairs").glob("*.png"))
    pos_names = sorted(p.name for p in (out_dir / "positives_pairs").glob("*.png"))
    assert len(neg_names) == 3
    assert any("vidA_001_f0001" in n for n in neg_names)
    assert not any("vidA_001_f0002" in n or "vidA_001_f0003" in n for n in neg_names)
    assert any("vidB_002_f0001" in n for n in neg_names)
    assert any("vidC_003_f0001" in n for n in neg_names)
    assert len(pos_names) == 2
    assert any("posA_010" in n for n in pos_names)
    assert any("posB_011" in n for n in pos_names)