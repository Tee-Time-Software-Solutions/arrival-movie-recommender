"""
Closed-loop ALS training: optional DB swipe export, then full implicit ALS pipeline.

Environment:
  RECOMMENDER_ARTIFACT_VERSION — if set, writes embeddings under artifacts/<version>/
  APP_USER_ID_OFFSET — namespace for app users vs MovieLens (export + metadata)
  DB_* — required for export unless --skip-export (same as scripts/export_swipes.py)

Run from backend/:
  uv run python -m movie_recommender.services.recommender.learning.offline_pipelines.closed_loop_pipeline
  uv run python -m movie_recommender.services.recommender.learning.offline_pipelines.closed_loop_pipeline --skip-export
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_movies import (
    preprocess_movies,
)
from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import (
    preprocess_ratings,
)
from movie_recommender.services.recommender.data_processing.preprocessing.filtering import (
    run_filtering,
)
from movie_recommender.services.recommender.data_processing.preprocessing.prune_movies import (
    prune_movies,
)
from movie_recommender.services.recommender.data_processing.split import run_split
from movie_recommender.services.recommender.learning.build_matrix import (
    build_sparse_matrix,
)
from movie_recommender.services.recommender.learning.evaluate import evaluate
from movie_recommender.services.recommender.learning.train_als import train
from movie_recommender.services.recommender.data_processing.swipe_export import (
    SWIPES_FROM_DB_FILENAME,
)
from movie_recommender.services.recommender.paths_dev import DATA_RAW

BACKEND_ROOT = Path(__file__).resolve().parents[6]
EXPORT_SCRIPT = BACKEND_ROOT / "scripts" / "export_swipes.py"


def _run_export_subprocess(app_user_id_offset: int | None) -> None:
    cmd = [sys.executable, str(EXPORT_SCRIPT)]
    if app_user_id_offset is not None:
        cmd.extend(["--app-user-id-offset", str(app_user_id_offset)])
    env = os.environ.copy()
    subprocess.run(cmd, cwd=str(BACKEND_ROOT), env=env, check=True)


def run_closed_loop(
    *,
    skip_export: bool = False,
    app_user_id_offset: int | None = None,
) -> None:
    start_time = time.time()
    print("\n========== CLOSED-LOOP ALS PIPELINE START ==========\n")
    ver = os.getenv("RECOMMENDER_ARTIFACT_VERSION", "").strip()
    if ver:
        print(f"Artifact version: {ver} (outputs under artifacts/{ver}/)")
    else:
        print("Artifact version: (default) outputs under artifacts/")

    if not skip_export:
        print("\nStep 0: Exporting swipes from Postgres...")
        _run_export_subprocess(app_user_id_offset)
        export_path = DATA_RAW / SWIPES_FROM_DB_FILENAME
        print(f"  Raw export: {export_path}")
        print(
            "  Step 2 will print MovieLens vs app swipes vs unified merge (row counts, prefs, ranges)."
        )
    else:
        print("\nStep 0: Skipping swipe export (--skip-export)")

    print("\nStep 1: Preprocessing movies...")
    preprocess_movies()

    print("\nStep 2: Preprocessing ratings (MovieLens + optional swipes)...")
    preprocess_ratings()

    print("\nStep 3: Filtering sparse users/movies...")
    run_filtering()

    print("\nStep 4: Pruning movie metadata...")
    prune_movies()

    print("\nStep 5: Chronological split...")
    run_split()

    print("\nStep 6: Building sparse matrix...")
    build_sparse_matrix()

    print("\nStep 7: Training ALS...")
    train()

    print("\nStep 8: Evaluating model...")
    evaluate()

    total_time = time.time() - start_time
    print("\n========== CLOSED-LOOP PIPELINE COMPLETE ==========")
    print(f"Total runtime: {total_time / 60:.2f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export swipes (optional) and run full ALS offline pipeline"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Do not call scripts/export_swipes.py (use existing data/raw/swipes_from_db.parquet or MovieLens-only)",
    )
    parser.add_argument(
        "--app-user-id-offset",
        type=int,
        default=None,
        help="Forwarded to export script when export runs (overrides APP_USER_ID_OFFSET env)",
    )
    args = parser.parse_args()
    run_closed_loop(
        skip_export=args.skip_export,
        app_user_id_offset=args.app_user_id_offset,
    )


if __name__ == "__main__":
    main()
