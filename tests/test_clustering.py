import pytest
import numpy as np
from src.processing.clustering import k_center_greedy_threshold

def test_k_center_greedy_empty():
    embeddings = np.empty((0, 512))
    
    result = k_center_greedy_threshold(embeddings, 0.5)
    
    assert result == []


def test_k_center_greedy_single():
    embeddings = np.array([[1.0, 0.0]])
    
    result = k_center_greedy_threshold(embeddings, 0.5)
    
    assert result == [0]


def test_k_center_greedy_identical():
    # Two identical L2-normalized vectors (cosine distance is 0.0)
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
    
    result = k_center_greedy_threshold(embeddings, 0.05)
    
    assert result == [0]


def test_k_center_greedy_orthogonal():
    # Two orthogonal L2-normalized vectors (cosine distance is 1.0)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    result = k_center_greedy_threshold(embeddings, 0.5)
    
    assert result == [0, 1]



def test_zero_theta_selects_each_frame_once():
    embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    assert k_center_greedy_threshold(
        embeddings, 0.0,
    ) == [0, 2, 1]


@pytest.mark.parametrize("theta", [-0.1, np.nan, np.inf])
def test_invalid_theta_fails(theta):
    with pytest.raises(ValueError, match="theta"):
        k_center_greedy_threshold(
            np.array([[1.0, 0.0]]),
            theta,
        )