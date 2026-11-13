from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct


def load_embedding_file(path: Path) -> dict[str, np.ndarray]:
    allowed = [
        (_reconstruct, "numpy._core.multiarray._reconstruct"),
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float16)),
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]

    with torch.serialization.safe_globals(allowed):
        data = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(data, dict):
        raise ValueError(f"Expected an embedding dictionary in {path}")

    vectors = {}
    for name, value in data.items():
        if not isinstance(name, str):
            raise ValueError(f"Invalid embedding key in {path}")

        if isinstance(value, torch.Tensor):
            value = value.detach().numpy()

        if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
            raise ValueError(f"Expected a floating-point embedding for {name} in {path}")

        vectors[name] = value

    return vectors