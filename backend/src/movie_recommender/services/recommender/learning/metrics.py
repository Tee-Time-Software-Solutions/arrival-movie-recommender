from __future__ import annotations

import numpy as np


def dcg_at_k(relevance: list[int] | list[float]) -> float:
    """
    Discounted Cumulative Gain at K for a ranked relevance list.
    """
    return float(sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)))
