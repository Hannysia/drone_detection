import pickle

import numpy as np
import pytest
import torch

from src.utils.embedding_io import load_embedding_file


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