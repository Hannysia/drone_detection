import pytest
import torch
from src.processing.diagnostics import find_sppf_layer
import logging
import sys

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