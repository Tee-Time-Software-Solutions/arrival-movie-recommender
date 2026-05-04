"""Leave-one-out eval: ALS-only vs ALS+KG graph rerank.

Three configurations compared per user:
  - als_only        : pure ALS scoring (no diversity, no graph rerank).
  - als_diversity   : ALS + MMR diversity reranking.
  - als_kg          : ALS -> top-K shortlist -> graph blend -> MMR diversity.

Metrics (k=10 by default): Hit@k, Precision@k, Recall@k, NDCG@k.

Run requirements:
- `make dev-start` running (Postgres on :5432, Redis on :6379, Neo4j on :7687).
- ALS artifacts present in pipeline/artifacts/model_assets/ (built by
  `make recommender-train-als`, which dev-start runs as a prerequisite).
- The Neo4j graph populated with movie metadata (writer.py / KG seeders).
- `tmdb_id` present in `movies_filtered.parquet` (offline pipeline change).

Usage:
    cd backend && .venv/bin/python scripts/eval_kg_rerank_vs_baseline.py \\
        [--n-users 50] [--k 10] [--graph-weight 0.3] [--shortlist-k 200] \\
        [--output scripts/eval_results/kg_rerank.csv]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "env_config" / "synced" / ".env.dev")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
sys.path.insert(0, str(ROOT / "src"))

import math  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
from sqlalchemy import select  # noqa: E402

from movie_recommender.core.clients.neo4j import Neo4jClient  # noqa: E402
from movie_recommender.core.settings.main import AppSettings  # noqa: E402
from movie_recommender.database.engine import DatabaseEngine  # noqa: E402
from movie_recommender.database.models import movies, swipes  # noqa: E402
from movie_recommender.services.knowledge_graph.beacon import (  # noqa: E402
    ENTITY_MULTIPLIERS,
    RECENCY_DECAY,
    SWIPE_SCORES,
    BeaconEntry,
    BeaconMap,
    _get_movie_entities,
)
from movie_recommender.services.recommender.pipeline.online.artifacts import (  # noqa: E402
    load_model_artifacts,
)
from movie_recommender.services.recommender.pipeline.online.serving.graph_rerank import (  # noqa: E402
    als_shortlist,
    blend_scores,
    compute_graph_scores,
)
from movie_recommender.services.recommender.pipeline.online.serving.ranker import (  # noqa: E402
    rank_movie_ids,
    score_candidates,
    select_top_n,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (  # noqa: E402
    base_user_vector,
)
from movie_recommender.services.recommender.pipeline.offline.models.base.steps.fetch_app_swipes import (  # noqa: E402
    get_app_user_id_offset,
)

# Reuse helpers from the chatbot eval to keep metrics & user-eligibility logic
# identical across reports.
sys.path.insert(0, str(ROOT / "scripts"))
from eval_chatbot_vs_baseline import (  # noqa: E402
    fetch_eligible_users,
    fetch_user_swipes,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    title_for,
)


CONFIGS = ("als_only", "als_diversity", "als_kg")


async def beacon_map_excluding(
    neo4j_driver, session_factory, user_id: int, holdout_movie_id: int
) -> BeaconMap:
    """Build the beacon map from all swipes EXCEPT the holdout movie.

    Mirrors `build_beacon_map` (services/knowledge_graph/beacon.py) but filters
    out the holdout row before aggregation, so the held-out movie's directors,
    actors, etc. do not leak into the user's beacon and unfairly help the KG
    rerank find the answer.

    Does not write to Redis (eval-only snapshot).
    """
    async with session_factory() as db:
        result = await db.execute(
            select(
                swipes.c.movie_id,
                swipes.c.action_type,
                swipes.c.is_supercharged,
                swipes.c.created_at,
                movies.c.tmdb_id,
            )
            .join(movies, movies.c.id == swipes.c.movie_id)
            .where(
                swipes.c.user_id == user_id,
                swipes.c.movie_id != holdout_movie_id,
                movies.c.tmdb_id.isnot(None),
            )
            .order_by(swipes.c.created_at.desc())
        )
        rows = result.fetchall()

    if not rows:
        return {}

    beacon_map: BeaconMap = {}
    now = datetime.now(timezone.utc)

    for row in rows:
        score = SWIPE_SCORES.get((row.action_type, row.is_supercharged), 0.0)
        if score == 0.0:
            continue
        days_ago = (now - row.created_at.replace(tzinfo=timezone.utc)).days
        decay = math.pow(RECENCY_DECAY, days_ago)
        weighted_score = score * decay

        entities = await _get_movie_entities(neo4j_driver, row.tmdb_id)
        for entity_type, tmdb_id, name in entities:
            multiplier = ENTITY_MULTIPLIERS.get(entity_type, 0.5)
            key = (entity_type, tmdb_id)
            if key not in beacon_map:
                beacon_map[key] = BeaconEntry(
                    entity_type=entity_type,
                    tmdb_id=tmdb_id,
                    name=name,
                    weight=0.0,
                )
            beacon_map[key].weight += weighted_score * multiplier

    return beacon_map


def _tmdb_id_lookup(artifacts) -> dict[int, int]:
    return artifacts.movie_id_to_tmdb_id


async def _als_kg_recommendations(
    artifacts,
    user_vector: np.ndarray,
    seen_set: set[int],
    diversity_weight: float,
    graph_weight: float,
    shortlist_k: int,
    neo4j_driver,
    beacon_map: BeaconMap,
    k: int,
) -> list[int]:
    cand_ids, cand_embs, scores = score_candidates(
        model_artifacts=artifacts,
        user_vector=user_vector,
        seen_movie_ids=seen_set,
    )

    if (
        graph_weight > 0
        and beacon_map
        and len(cand_ids) > 0
        and artifacts.movie_id_to_tmdb_id
    ):
        movie_id_to_tmdb_id = artifacts.movie_id_to_tmdb_id
        shortlist_idx, shortlist_ids = als_shortlist(scores, cand_ids, shortlist_k)

        shortlist_tmdb_ids: list[int] = []
        idx_to_tmdb: list[tuple[int, int]] = []
        for pos, mid in enumerate(shortlist_ids.tolist()):
            tid = movie_id_to_tmdb_id.get(int(mid))
            if tid is not None:
                shortlist_tmdb_ids.append(tid)
                idx_to_tmdb.append((pos, tid))

        if shortlist_tmdb_ids:
            graph_dict = await compute_graph_scores(
                neo4j_driver, shortlist_tmdb_ids, beacon_map
            )
            if graph_dict:
                graph_array = np.zeros(len(shortlist_idx), dtype=np.float32)
                for pos, tid in idx_to_tmdb:
                    graph_array[pos] = graph_dict.get(tid, 0.0)
                blended = blend_scores(
                    scores[shortlist_idx], graph_array, graph_weight
                )
                scores[shortlist_idx] = blended

    return select_top_n(
        candidate_ids=cand_ids,
        candidate_embeddings=cand_embs,
        scores=scores,
        n=k,
        diversity_weight=diversity_weight,
    )


async def main(
    n_users: int,
    k: int,
    graph_weight: float,
    shortlist_k: int,
    output_path: Path,
    require_in_training: bool,
) -> None:
    settings = AppSettings()
    artifacts = load_model_artifacts()
    offset = get_app_user_id_offset()
    session_factory = DatabaseEngine().session_factory
    neo4j_driver = await Neo4jClient().get_async_driver()

    if not artifacts.movie_id_to_tmdb_id:
        print(
            "WARNING: artifacts have no movie_id -> tmdb_id mapping. "
            "Re-run the offline pipeline so movies_filtered.parquet carries tmdb_id."
        )

    eligible = await fetch_eligible_users(
        session_factory, min_likes=10, require_in_training=require_in_training
    )
    if not eligible:
        msg = "No eligible users with >= 10 likes"
        if require_in_training:
            msg += " AND in ALS training set"
        print(f"{msg}. Aborting.")
        return
    test_users = eligible[:n_users]
    print(
        f"Eval on {len(test_users)} users (k={k}, graph_weight={graph_weight}, "
        f"shortlist_k={shortlist_k}, diversity_weight={settings.app_logic.diversity_weight})."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    aggregates = {cfg: {"hit": 0.0, "prec": 0.0, "rec": 0.0, "ndcg": 0.0} for cfg in CONFIGS}

    for idx, user_id in enumerate(test_users, start=1):
        liked, all_swiped = await fetch_user_swipes(session_factory, user_id)
        if len(liked) < 2:
            print(f"  [{idx}/{len(test_users)}] user={user_id}: not enough likes — skip.")
            continue
        holdout = liked[-1]
        seen_set = set(all_swiped) - {holdout}

        ml_user_id = user_id + offset
        user_vector = base_user_vector(artifacts, ml_user_id)

        beacon_map: BeaconMap = {}
        try:
            beacon_map = await beacon_map_excluding(
                neo4j_driver, session_factory, user_id, holdout
            )
        except Exception as e:
            print(f"  [{idx}/{len(test_users)}] user={user_id}: beacon build failed: {e!r}")

        # 1. als_only
        als_only_recs = rank_movie_ids(
            n=k,
            model_artifacts=artifacts,
            user_vector=user_vector,
            seen_movie_ids=seen_set,
        )

        # 2. als_diversity
        als_div_recs = rank_movie_ids(
            n=k,
            model_artifacts=artifacts,
            user_vector=user_vector,
            seen_movie_ids=seen_set,
            diversity_weight=settings.app_logic.diversity_weight,
        )

        # 3. als_kg
        als_kg_recs = await _als_kg_recommendations(
            artifacts=artifacts,
            user_vector=user_vector,
            seen_set=seen_set,
            diversity_weight=settings.app_logic.diversity_weight,
            graph_weight=graph_weight,
            shortlist_k=shortlist_k,
            neo4j_driver=neo4j_driver,
            beacon_map=beacon_map,
            k=k,
        )

        per_cfg = {
            "als_only": als_only_recs,
            "als_diversity": als_div_recs,
            "als_kg": als_kg_recs,
        }

        row = {
            "user_id": user_id,
            "holdout_movie_id": holdout,
            "holdout_title": title_for(artifacts, holdout),
            "beacon_size": len(beacon_map),
        }
        for cfg, recs in per_cfg.items():
            metrics = {
                "hit": hit_at_k(recs, holdout),
                "prec": precision_at_k(recs, holdout, k),
                "rec": recall_at_k(recs, holdout),
                "ndcg": ndcg_at_k(recs, holdout),
            }
            for key, val in metrics.items():
                aggregates[cfg][key] += val
            row[f"{cfg}_recs"] = ",".join(map(str, recs))
            row[f"{cfg}_hit"] = metrics["hit"]
            row[f"{cfg}_prec"] = round(metrics["prec"], 4)
            row[f"{cfg}_recall"] = round(metrics["rec"], 4)
            row[f"{cfg}_ndcg"] = round(metrics["ndcg"], 4)

        rows.append(row)
        print(
            f"  [{idx}/{len(test_users)}] user={user_id} "
            f"holdout={title_for(artifacts, holdout)!r}  "
            f"als_only_ndcg={row['als_only_ndcg']:.3f} "
            f"als_kg_ndcg={row['als_kg_ndcg']:.3f} "
            f"beacon_size={len(beacon_map)}"
        )

    n = max(len(rows), 1)
    summary = {
        cfg: {key: round(val / n, 4) for key, val in m.items()}
        for cfg, m in aggregates.items()
    }

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "n_users": len(rows),
                "k": k,
                "graph_weight": graph_weight,
                "shortlist_k": shortlist_k,
                "diversity_weight": settings.app_logic.diversity_weight,
                "metrics": summary,
            },
            indent=2,
        )
    )

    print(f"\n=== Summary (mean over {n} users, k={k}) ===")
    print(f"             hit@{k}   prec@{k}   recall@{k}  ndcg@{k}")
    for cfg in CONFIGS:
        m = summary[cfg]
        print(
            f"  {cfg:13} {m['hit']:.3f}   {m['prec']:.3f}    "
            f"{m['rec']:.3f}      {m['ndcg']:.3f}"
        )
    print(f"\nWrote {output_path} and {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.3,
        help="Mixing fraction for ALS vs graph score in the blend (0..1).",
    )
    parser.add_argument(
        "--shortlist-k",
        type=int,
        default=200,
        help="Top-K ALS candidates to apply graph rerank to.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts" / "eval_results" / "eval_kg_rerank_vs_baseline.csv",
    )
    parser.add_argument(
        "--require-in-training",
        action="store_true",
        help="Only evaluate users whose ALS vector was trained.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.n_users,
            args.k,
            args.graph_weight,
            args.shortlist_k,
            args.output,
            args.require_in_training,
        )
    )
