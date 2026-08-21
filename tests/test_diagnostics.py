import pytest
from src.processing.diagnostics import find_sppf_layer

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