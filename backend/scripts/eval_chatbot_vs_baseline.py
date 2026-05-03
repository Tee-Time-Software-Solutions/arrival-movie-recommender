"""Leave-one-out eval: chatbot agent vs ALS baseline recommender.

Metrics taken from slides_corpus.md:
- Precision@k, Recall@k, NDCG@k, HitRate@k    (SAR session, lines 270-271)
- leave-one-out evaluation                    (NCF session, line 449)

Pipeline (per user):
  1. Hold out the most recent liked movie ("test item").
  2. Mark every prior swiped movie as "seen" (excluded from candidates).
  3. ALS baseline:  rank_movie_ids() with the user's ALS-trained vector.
  4. Agent:         astream_events on a recommendation prompt; collect
                    movie_db_id values from search_movies tool outputs.
  5. Compute Hit/Precision/Recall/NDCG @K against {test item}.

Run requirements:
- `make dev-start` running (Postgres on :5432, Redis on :6379 — both exposed
  in deployment/docker-compose.yml).
- ALS artifacts present in pipeline/artifacts/model_assets/  (built by
  `make recommender-train-als`, which dev-start runs as a prerequisite).
- OPENROUTER_API_KEY in env_config/synced/.env.dev.

Usage:
    cd backend && .venv/bin/python scripts/eval_chatbot_vs_baseline.py \\
        [--n-users 10] [--k 10] [--output scripts/eval_results/eval.csv]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
load_dotenv(ROOT / "env_config" / "synced" / ".env.dev", override=False)
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select, func, desc  # noqa: E402

from movie_recommender.database.engine import DatabaseEngine  # noqa: E402
from movie_recommender.database.models import swipes, movies as movies_table  # noqa: E402
from movie_recommender.services.chatbot.agent import build_agent  # noqa: E402
from movie_recommender.services.recommender.pipeline.online.artifacts import (  # noqa: E402
    load_model_artifacts,
)
from movie_recommender.services.recommender.pipeline.online.serving.ranker import (  # noqa: E402
    rank_movie_ids,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (  # noqa: E402
    base_user_vector,
)
from movie_recommender.services.recommender.pipeline.offline.models.base.steps.fetch_app_swipes import (  # noqa: E402
    get_app_user_id_offset,
)


AGENT_PROMPT = (
    "Recommend 10 movies for me to watch tonight. "
    "Use my taste profile to personalise the suggestions. "
    "Return a single ranked list of 10 distinct movies."
)


# ---- DB helpers ------------------------------------------------------------


async def fetch_eligible_users(
    session_factory, min_likes: int, require_in_training: bool
) -> list[int]:
    """App user_ids with >= min_likes. If require_in_training, restrict to those
    whose `+offset` id is in the ALS training set (so they have a trained vector
    rather than a cold-start mean)."""
    artifacts = load_model_artifacts()
    offset = get_app_user_id_offset()
    trained_app_ids = {
        ml_uid - offset
        for ml_uid in artifacts.user_id_to_index
        if ml_uid >= offset
    }

    async with session_factory() as db:
        rows = await db.execute(
            select(swipes.c.user_id, func.count().label("n"))
            .where(swipes.c.action_type == "like")
            .group_by(swipes.c.user_id)
            .having(func.count() >= min_likes)
            .order_by(desc("n"))
        )
        all_ids = [r.user_id for r in rows]
    if require_in_training:
        return [uid for uid in all_ids if uid in trained_app_ids]
    return all_ids


async def fetch_user_swipes(session_factory, user_id: int) -> tuple[list[int], list[int]]:
    """Return (liked_ids_oldest_first, all_swiped_ids)."""
    async with session_factory() as db:
        likes = await db.execute(
            select(swipes.c.movie_id)
            .where(swipes.c.user_id == user_id, swipes.c.action_type == "like")
            .order_by(swipes.c.created_at.asc())
        )
        all_swiped = await db.execute(
            select(swipes.c.movie_id.distinct()).where(swipes.c.user_id == user_id)
        )
    liked = [r.movie_id for r in likes]
    all_ids = [r.movie_id for r in all_swiped]
    return liked, all_ids


# ---- Agent invocation ------------------------------------------------------


async def agent_recommend(session_factory, user_id: int, k: int) -> tuple[list[int], dict[str, Any]]:
    """Run the agent on a recommendation prompt and extract its top-k movie_db_ids.

    Returns (movie_db_ids, debug_info). Order = order of appearance in tool outputs.
    """
    agent = build_agent(session_factory, user_id)
    tool_calls: list[dict] = []
    final_text = ""

    async for event in agent.astream_events(
        {"messages": [("user", AGENT_PROMPT)]}, version="v2"
    ):
        kind = event["event"]
        if kind == "on_tool_end":
            name = event.get("name", "")
            raw = event["data"].get("output")
            output_str = raw.content if hasattr(raw, "content") else str(raw)
            tool_calls.append({"name": name, "output": output_str})
        elif kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            content = getattr(chunk, "content", "") if chunk is not None else ""
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            if content:
                final_text += content

    seen: set[int] = set()
    ordered: list[int] = []
    for call in tool_calls:
        if call["name"] != "search_movies":
            continue
        try:
            payload = json.loads(call["output"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, list):
            continue
        for movie in payload:
            mid = movie.get("movie_db_id") if isinstance(movie, dict) else None
            if isinstance(mid, int) and mid not in seen:
                seen.add(mid)
                ordered.append(mid)
            if len(ordered) >= k:
                break
        if len(ordered) >= k:
            break

    return ordered[:k], {
        "tool_calls_count": len(tool_calls),
        "search_calls": sum(1 for c in tool_calls if c["name"] == "search_movies"),
        "final_text_chars": len(final_text),
    }


# ---- Metrics ---------------------------------------------------------------


def hit_at_k(recs: list[int], holdout: int) -> int:
    return int(holdout in recs)


def precision_at_k(recs: list[int], holdout: int, k: int) -> float:
    return (1.0 / k) if holdout in recs else 0.0


def recall_at_k(recs: list[int], holdout: int) -> float:
    return 1.0 if holdout in recs else 0.0


def ndcg_at_k(recs: list[int], holdout: int) -> float:
    """LOO-NDCG: log2(2) / log2(rank+1) if holdout at position rank (1-indexed), else 0.

    Single relevant item -> IDCG = 1, so NDCG = DCG.
    """
    for i, m in enumerate(recs, start=1):
        if m == holdout:
            return 1.0 / math.log2(i + 1)
    return 0.0


def title_for(artifacts, movie_id: int) -> str:
    return artifacts.movie_id_to_title.get(int(movie_id), f"#{movie_id}")


# ---- Main ------------------------------------------------------------------


async def main(n_users: int, k: int, output_path: Path, require_in_training: bool) -> None:
    artifacts = load_model_artifacts()
    offset = get_app_user_id_offset()
    session_factory = DatabaseEngine().session_factory

    eligible = await fetch_eligible_users(
        session_factory, min_likes=10, require_in_training=require_in_training,
    )
    if not eligible:
        msg = "No eligible users with >= 10 likes"
        if require_in_training:
            msg += " AND in ALS training set"
        print(f"{msg}. Aborting.")
        return
    test_users = eligible[:n_users]
    print(f"Eval on {len(test_users)} users (k={k}).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    aggregates = {
        "baseline": {"hit": 0.0, "prec": 0.0, "rec": 0.0, "ndcg": 0.0},
        "agent": {"hit": 0.0, "prec": 0.0, "rec": 0.0, "ndcg": 0.0},
    }

    for idx, user_id in enumerate(test_users, start=1):
        liked, all_swiped = await fetch_user_swipes(session_factory, user_id)
        if len(liked) < 2:
            print(f"  [{idx}/{len(test_users)}] user={user_id}: not enough likes — skip.")
            continue
        holdout = liked[-1]  # most recent like
        seen_set = set(all_swiped) - {holdout}

        ml_user_id = user_id + offset
        baseline_recs = rank_movie_ids(
            n=k,
            model_artifacts=artifacts,
            user_vector=base_user_vector(artifacts, ml_user_id),
            seen_movie_ids=seen_set,
        )

        try:
            agent_recs, debug = await agent_recommend(session_factory, user_id, k)
        except Exception as e:
            print(f"  [{idx}/{len(test_users)}] user={user_id}: agent error: {e!r}")
            agent_recs, debug = [], {"error": repr(e)}

        b = {
            "hit": hit_at_k(baseline_recs, holdout),
            "prec": precision_at_k(baseline_recs, holdout, k),
            "rec": recall_at_k(baseline_recs, holdout),
            "ndcg": ndcg_at_k(baseline_recs, holdout),
        }
        a = {
            "hit": hit_at_k(agent_recs, holdout),
            "prec": precision_at_k(agent_recs, holdout, k),
            "rec": recall_at_k(agent_recs, holdout),
            "ndcg": ndcg_at_k(agent_recs, holdout),
        }
        for src, m in (("baseline", b), ("agent", a)):
            for key, val in m.items():
                aggregates[src][key] += val

        rows.append({
            "user_id": user_id,
            "holdout_movie_id": holdout,
            "holdout_title": title_for(artifacts, holdout),
            "baseline_recs": ",".join(map(str, baseline_recs)),
            "agent_recs": ",".join(map(str, agent_recs)),
            "agent_first_titles": " | ".join(title_for(artifacts, m) for m in agent_recs[:3]),
            "baseline_hit": b["hit"], "baseline_prec": round(b["prec"], 4),
            "baseline_recall": b["rec"], "baseline_ndcg": round(b["ndcg"], 4),
            "agent_hit": a["hit"], "agent_prec": round(a["prec"], 4),
            "agent_recall": a["rec"], "agent_ndcg": round(a["ndcg"], 4),
            "agent_search_calls": debug.get("search_calls", 0),
        })
        print(
            f"  [{idx}/{len(test_users)}] user={user_id} "
            f"holdout={title_for(artifacts, holdout)!r}  "
            f"baseline_hit={b['hit']} agent_hit={a['hit']} "
            f"baseline_ndcg={b['ndcg']:.3f} agent_ndcg={a['ndcg']:.3f}"
        )

    n = max(len(rows), 1)
    summary = {
        src: {key: round(val / n, 4) for key, val in m.items()}
        for src, m in aggregates.items()
    }

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "n_users": len(rows), "k": k, "metrics": summary,
    }, indent=2))

    print("\n=== Summary (mean over users) ===")
    print(f"           hit@{k}   prec@{k}   recall@{k}  ndcg@{k}")
    for src in ("baseline", "agent"):
        m = summary[src]
        print(f"  {src:8} {m['hit']:.3f}   {m['prec']:.3f}    {m['rec']:.3f}      {m['ndcg']:.3f}")
    print(f"\nWrote {output_path} and {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts" / "eval_results" / "eval_chatbot_vs_baseline.csv",
    )
    parser.add_argument(
        "--require-in-training",
        action="store_true",
        help="Only evaluate users whose ALS vector was trained (vs cold-start mean).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.n_users, args.k, args.output, args.require_in_training))
