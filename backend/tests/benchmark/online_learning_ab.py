
"""
Run:
    cd backend && uv run python -m tests.benchmark.online_learning_ab
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
    load_model_artifacts,
)
from movie_recommender.services.recommender.pipeline.online.learning.feedback import (
    apply_feedback_update,
)
from movie_recommender.services.recommender.pipeline.online.serving.ranker import (
    rank_movie_ids,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (
    cold_start_vector,
)
from movie_recommender.services.recommender.utils.schema import load_config

K = 10
LEARNING_RATE = 0.5
NORM_CAP = 10.0
# Bar chart / stat test: full-history replay for users with >= MIN_VAL events
MIN_VAL = 20
MIN_TEST = 3
# Time curve: balanced panel of users with >= PANEL_MIN events, all truncated
# to PANEL_EVENTS so every checkpoint has the same n.
PANEL_MIN = 50
PANEL_EVENTS = 50
CURVE_STEPS = (0, 5, 10, 15, 20, 30, 40, 50)
MMR_DIVERSITY_WEIGHT = 0.3

REPO_ROOT = Path(__file__).resolve().parents[2].parent
ASSETS_OUT = REPO_ROOT / "docs" / "assets"


def preference_to_swipe(pref: int) -> tuple[SwipeAction, bool]:
    if pref >= 2:
        return SwipeAction.LIKE, True
    if pref == 1:
        return SwipeAction.LIKE, False
    if pref == 0:
        return SwipeAction.SKIP, False
    if pref == -1:
        return SwipeAction.DISLIKE, False
    return SwipeAction.DISLIKE, True


def dcg(rels: list[int]) -> float:
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(rels)))


def metrics_at_k(recommended: list[int], truth: set[int]) -> tuple[float, float, float]:
    if not recommended:
        return 0.0, 0.0, 0.0
    rels = [1 if m in truth else 0 for m in recommended]
    hits = sum(rels)
    precision = hits / len(recommended)
    recall = hits / max(len(truth), 1)
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    ndcg = dcg(rels) / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def intra_list_diversity(
    recommended: list[int], artifacts: RecommenderArtifacts
) -> float:
    if len(recommended) < 2:
        return 0.0
    idx = [
        artifacts.movie_id_to_index[m]
        for m in recommended
        if m in artifacts.movie_id_to_index
    ]
    if len(idx) < 2:
        return 0.0
    embs = artifacts.movie_embeddings[idx]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embs / norms
    sim = unit @ unit.T
    n = len(idx)
    upper = sim[np.triu_indices(n, k=1)]
    return float(1.0 - upper.mean())


def replay_to_checkpoints(
    artifacts: RecommenderArtifacts,
    val_events: pd.DataFrame,
    checkpoints: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Replay events on a cold-start vector. Return a snapshot of the user
    vector at each checkpoint (number of events seen). Includes 0 (pre-replay)."""
    vec = cold_start_vector(artifacts).copy()
    snapshots: dict[int, np.ndarray] = {}
    if 0 in checkpoints:
        snapshots[0] = vec.copy()
    target = max(checkpoints)
    for i, (_, row) in enumerate(val_events.iterrows(), start=1):
        if i > target:
            break
        action, supercharged = preference_to_swipe(int(row.preference))
        updated = apply_feedback_update(
            model_artifacts=artifacts,
            user_vector=vec,
            movie_id=int(row.movie_id),
            interaction_type=action,
            is_supercharged=supercharged,
            learning_rate=LEARNING_RATE,
            norm_cap=NORM_CAP,
        )
        if updated is not None:
            vec = updated
        if i in checkpoints:
            snapshots[i] = vec.copy()
    return snapshots


def main() -> None:
    print("Loading model artifacts and splits...")
    artifacts = load_model_artifacts()
    config = load_config()
    splits_dir = config.data_dirs.splits_dir

    train = pd.read_parquet(splits_dir / "train.parquet")
    val = pd.read_parquet(splits_dir / "val.parquet")
    test = pd.read_parquet(splits_dir / "test.parquet")

    train_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    val_by_user = {uid: g.sort_values("timestamp") for uid, g in val.groupby("user_id")}
    test_truth = (
        test[test["preference"] > 0].groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    eligible = sorted(
        u
        for u in test_truth
        if u in val_by_user
        and len(val_by_user[u]) >= MIN_VAL
        and len(test_truth[u]) >= MIN_TEST
    )
    panel_users = sum(1 for u in eligible if len(val_by_user[u]) >= PANEL_MIN)
    print(
        f"Eligible users: {len(eligible)} (full-replay arms); "
        f"{panel_users} of those qualify for the {PANEL_EVENTS}-event balanced panel"
    )

    static_p, static_r, static_n, static_div = [], [], [], []
    online_p, online_r, online_n, online_div = [], [], [], []
    curve_buckets: dict[int, list[float]] = {step: [] for step in CURVE_STEPS}

    for user_id in eligible:
        # In production, train + val swipes are both in the seen-set; mirror that
        # here for a fair comparison across arms.
        seen = set(train_seen.get(user_id, set()))
        seen.update(val_by_user[user_id]["movie_id"].astype(int).tolist())
        truth = test_truth[user_id]
        truth_in_index = {m for m in truth if m in artifacts.movie_id_to_index}
        if not truth_in_index:
            continue

        events = val_by_user[user_id]
        snaps_full = replay_to_checkpoints(artifacts, events, (0, len(events)))
        vec_static = snaps_full[0]
        vec_online = snaps_full[len(events)]

        rec_static = rank_movie_ids(
            n=K,
            model_artifacts=artifacts,
            user_vector=vec_static,
            seen_movie_ids=seen,
        )
        p, r, n = metrics_at_k(rec_static, truth_in_index)
        static_p.append(p)
        static_r.append(r)
        static_n.append(n)
        static_div.append(intra_list_diversity(rec_static, artifacts))

        rec_online = rank_movie_ids(
            n=K,
            model_artifacts=artifacts,
            user_vector=vec_online,
            seen_movie_ids=seen,
            diversity_weight=MMR_DIVERSITY_WEIGHT,
        )
        p, r, n = metrics_at_k(rec_online, truth_in_index)
        online_p.append(p)
        online_r.append(r)
        online_n.append(n)
        online_div.append(intra_list_diversity(rec_online, artifacts))

        # Time-curve panel — only users with enough events for the truncation
        if len(events) >= PANEL_MIN:
            panel_snaps = replay_to_checkpoints(artifacts, events, CURVE_STEPS)
            for step in CURVE_STEPS:
                vec = panel_snaps.get(step)
                if vec is None:
                    continue
                rec = rank_movie_ids(
                    n=K,
                    model_artifacts=artifacts,
                    user_vector=vec,
                    seen_movie_ids=seen,
                    diversity_weight=MMR_DIVERSITY_WEIGHT if step > 0 else 0.0,
                )
                _, _, ndcg_chk = metrics_at_k(rec, truth_in_index)
                curve_buckets[step].append(ndcg_chk)

    rows = [
        (
            "Static (cold-start, no online learning)",
            static_p,
            static_r,
            static_n,
            static_div,
        ),
        (
            "Online (vector update + MMR diversity)",
            online_p,
            online_r,
            online_n,
            online_div,
        ),
    ]

    print()
    print(f"=== A/B benchmark — K={K}, n_users={len(static_n)} ===")
    header = f"{'Arm':<40} {'P@K':>8} {'R@K':>8} {'NDCG@K':>9} {'Div@K':>8}"
    print(header)
    print("-" * len(header))
    for name, ps, rs, ns, ds in rows:
        div = f"{np.mean(ds):.4f}" if ds else "    -   "
        print(
            f"{name:<40} {np.mean(ps):>8.4f} {np.mean(rs):>8.4f} "
            f"{np.mean(ns):>9.4f} {div:>8}"
        )

    t_on, p_on = stats.ttest_rel(online_n, static_n)
    print(
        f"\nPaired t-test online vs static (NDCG@{K}): "
        f"t={t_on:.3f}, p={p_on:.2e}, n={len(online_n)}"
    )

    ASSETS_OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    arms = [name for name, *_ in rows]
    metric_names = ["Precision@K", "Recall@K", "NDCG@K"]
    means = np.array(
        [[np.mean(ps), np.mean(rs), np.mean(ns)] for _, ps, rs, ns, _ in rows]
    )
    x = np.arange(len(arms))
    width = 0.25
    for i, mn in enumerate(metric_names):
        ax.bar(x + (i - 1) * width, means[:, i], width, label=mn)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Static vs online (K={K}, n={len(online_n)}, paired t-test p={p_on:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_OUT / "ab_static_vs_online.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = sorted(curve_buckets.keys())
    panel_n = len(curve_buckets[xs[0]]) if curve_buckets[xs[0]] else 0
    ys = [np.mean(curve_buckets[x]) if curve_buckets[x] else np.nan for x in xs]
    ax.plot(xs, ys, marker="o", label="Online learning (balanced panel)")
    static_panel = curve_buckets[0]
    if static_panel:
        ax.axhline(
            np.mean(static_panel),
            color="gray",
            linestyle="--",
            label="Static baseline (cold-start, panel mean)",
        )
    ax.set_xlabel("Replayed feedback events")
    ax.set_ylabel(f"Mean NDCG@{K}")
    ax.set_title(
        f"NDCG@{K} over time — balanced panel (n={panel_n}, "
        f"each truncated to {PANEL_EVENTS} events)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_OUT / "ndcg_over_time.png", dpi=150)
    plt.close(fig)

    print()
    print(f"=== Time curve (balanced panel, n={panel_n}) ===")
    for x in xs:
        bucket = curve_buckets[x]
        if bucket:
            print(f"  after {x:>3} events: NDCG@{K}={np.mean(bucket):.4f}")

    print(f"\nWrote PNGs to {ASSETS_OUT}")


if __name__ == "__main__":
    main()
