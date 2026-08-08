import numpy as np

def k_center_greedy_threshold(embeddings: np.ndarray, theta: float) -> list[int]:
    """
    Performs k-Center Greedy selection based on a distance threshold.
    Returns a list of indices representing the selected diverse frames.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    dists = 1.0 - sim

    selected_centers = [0]
    min_dists = dists[0].copy()

    while True:
        farthest_idx = np.argmax(min_dists)
        max_dist = min_dists[farthest_idx]

        if max_dist < theta:
            break

        selected_centers.append(farthest_idx)
        min_dists = np.minimum(min_dists, dists[farthest_idx])

    return selected_centers