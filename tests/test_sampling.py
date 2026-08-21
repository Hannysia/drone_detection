import pytest
from src.processing.sampling import select_indices

def test_select_indices_empty():
    result = select_indices(0, 1, None)
    assert result == set()


def test_select_indices_keep_all():
    result = select_indices(5, 1, None)
    assert result == {0, 1, 2, 3, 4}


def test_select_indices_every_n_filtering():
    # Keep every 3rd frame from 10 frames
    result = select_indices(10, 3, None)
    assert result == {0, 3, 6, 9}


def test_select_indices_cap_limiting_uniform():
    # n_frames (10) is greater than cap (5) -> uniformly sample 5 frame indices
    result = select_indices(10, 1, 5)
    
    # np.linspace(0, 9, 5, dtype=int) yields: [0, 2, 4, 6, 9]
    assert result == {0, 2, 4, 6, 9}