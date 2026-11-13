import sys
from types import SimpleNamespace
from unittest.mock import Mock


def test_pipeline_returns_weights_without_test_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EXTRACTED_DATA_PATH", str(tmp_path))

    from src.processing import training

    monkeypatch.setattr(training, "EXTRACTED_DATA_DIR", tmp_path)

    yaml_path = tmp_path / "dataset.yaml"
    weights = tmp_path / "best.pt"
    fit = Mock(return_value=weights)
    evaluate = Mock()

    monkeypatch.setattr(
        training, "prepare_yolo_dataset", Mock(return_value=yaml_path)
    )
    monkeypatch.setattr(training, "run_yolo_training", fit)
    monkeypatch.setitem(
        sys.modules,
        "src.processing.evaluation",
        SimpleNamespace(evaluate_model=evaluate),
    )

    assert training.run_full_pipeline(False, "example") == weights
    fit.assert_called_once_with(yaml_path, "example")
    evaluate.assert_not_called()