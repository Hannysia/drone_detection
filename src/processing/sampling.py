import numpy as np

def select_indices(n_frames: int, every_n: int, cap: int | None) -> set[int]:
    """
    Returns a set of frame indices that should be KEPT.

    Logic:
    1. If cap is set and n_frames > cap — uniformly sample 'cap' indices.
    2. If every_n is set — keep every every_n-th frame.
    3. Otherwise, keep all frames.
    """
    if n_frames == 0:
        return set()

    if cap is not None and n_frames > cap:
        indices = np.linspace(0, n_frames - 1, cap, dtype=int)
        return set(indices.tolist())

    if every_n is not None and every_n > 1:
        return {i for i in range(n_frames) if i % every_n == 0}

    return set(range(n_frames))