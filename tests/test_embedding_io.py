import pickle
import numpy as np
import pytest
import torch
from src.utils.embedding_io import load_embedding_file, validate_embedding_matrix

class UnsupportedRecord:
    pass


def test_reads_existing_numpy_embedding_format(tmp_path):
    path = tmp_path / "embeddings.pt"
    vector = np.array([0.25, 0.75], dtype=np.float32)
    torch.save({"example.jpg": vector}, path)

    result = load_embedding_file(path)

    assert set(result) == {"example.jpg"}
    np.testing.assert_array_equal(result["example.jpg"], vector)
    assert result["example.jpg"].dtype == vector.dtype


def test_rejects_unlisted_class(tmp_path):
    path = tmp_path / "unsupported.pt"
    torch.save({"example.jpg": UnsupportedRecord()}, path)

    with pytest.raises(pickle.UnpicklingError):
        load_embedding_file(path)


def test_validate_embedding_matrix_accepts_unit_float_rows():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    validate_embedding_matrix(matrix, "demo_video")


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([1.0, 0.0]),
        np.empty((1, 0), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.array([[1, 0]]),
        np.array([[np.nan, 0.0]]),
        np.array([[np.inf, 0.0]]),
        np.array([[2.0, 0.0]]),
    ],
)
def test_validate_embedding_matrix_rejects_invalid_inputs(matrix):
    with pytest.raises(ValueError, match="L2-normalized embeddings for ctx"):
        validate_embedding_matrix(matrix, "ctx")