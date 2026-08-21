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