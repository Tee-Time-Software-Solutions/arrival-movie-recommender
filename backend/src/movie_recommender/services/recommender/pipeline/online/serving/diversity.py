import numpy as np


def mmr_rerank(
    scores: np.ndarray,
    embeddings: np.ndarray,
    top_k: int,
    lambda_mmr: float,
) -> np.ndarray:
    """Maximal Marginal Relevance re-ranking.

    Selects ``top_k`` indices that balance relevance (``scores``) against
    diversity (1 - max cosine similarity to the already-selected set).
    ``lambda_mmr`` in [0, 1]: 1.0 collapses to pure relevance, 0.0 to pure
    diversity. Returns the selected indices in MMR order.
    """
    n = len(scores)
    k = min(top_k, n)
    if k == 0:
        return np.empty(0, dtype=np.int64)

    norms = np.linalg.norm(embeddings, axis=1)
    safe = norms > 0
    unit = np.zeros_like(embeddings)
    unit[safe] = embeddings[safe] / norms[safe, None]

    selected: list[int] = []
    remaining = np.ones(n, dtype=bool)
    max_sim = np.full(n, -np.inf, dtype=np.float32)

    first = int(np.argmax(scores))
    selected.append(first)
    remaining[first] = False
    max_sim = unit @ unit[first]

    while len(selected) < k:
        mmr_scores = lambda_mmr * scores - (1.0 - lambda_mmr) * max_sim
        mmr_scores = np.where(remaining, mmr_scores, -np.inf)
        nxt = int(np.argmax(mmr_scores))
        selected.append(nxt)
        remaining[nxt] = False
        sim_to_new = unit @ unit[nxt]
        max_sim = np.maximum(max_sim, sim_to_new)

    return np.array(selected, dtype=np.int64)
