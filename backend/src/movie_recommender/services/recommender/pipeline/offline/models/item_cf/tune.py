from __future__ import annotations

import datetime
import itertools
import json
import time
from pathlib import Path
from typing import Any

import movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.matrix as matrix
import movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.metrics as metrics
import movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.train_item_cf as train_item_cf
from movie_recommender.services.recommender.utils.schema import Config, load_config


def _item_cf_config_as_dict(config: Config) -> dict[str, Any]:
    item_cf = config.models.item_cf
    return {
        "similarity": item_cf.similarity,
        "top_k_neighbors": item_cf.top_k_neighbors,
        "min_similarity": item_cf.min_similarity,
        "use_positive_only": item_cf.use_positive_only,
        "normalize_scores": item_cf.normalize_scores,
        "min_co_raters": item_cf.min_co_raters,
        "similarity_shrinkage": item_cf.similarity_shrinkage,
        "neighbor_weight_power": item_cf.neighbor_weight_power,
        "relevance_preference_threshold": item_cf.relevance_preference_threshold,
    }


def _apply_item_cf_overrides(config: Config, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        setattr(config.models.item_cf, key, value)


def _run_single_experiment(
    config: Config,
    label: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original = _item_cf_config_as_dict(config)
    try:
        if overrides:
            _apply_item_cf_overrides(config, overrides)

        active_config = _item_cf_config_as_dict(config)
        started_at = time.perf_counter()
        train_item_cf.run(config)
        metrics.run(config)
        elapsed_seconds = time.perf_counter() - started_at

        metrics_path = config.data_dirs.model_assets_dir / "item_cf_metrics.json"
        with open(metrics_path) as file_obj:
            metrics_report = json.load(file_obj)

        metric_values = metrics_report.get("metrics", {})
        return {
            "label": label,
            "config": active_config,
            "runtime_seconds": round(elapsed_seconds, 4),
            "num_users_evaluated": metrics_report.get("num_users_evaluated", 0),
            "precision@10": float(metric_values.get("precision@10", 0.0)),
            "recall@10": float(metric_values.get("recall@10", 0.0)),
            "ndcg@10": float(metric_values.get("ndcg@10", 0.0)),
        }
    finally:
        _apply_item_cf_overrides(config, original)


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (row["ndcg@10"], row["recall@10"], row["precision@10"]),
        reverse=True,
    )


def _print_summary(title: str, records: list[dict[str, Any]], top_n: int = 10) -> None:
    print(f"\n=== {title} ===")
    for idx, row in enumerate(records[:top_n], start=1):
        print(
            f"{idx:02d}. {row['label']:<14} ndcg={row['ndcg@10']:.4f} "
            f"recall={row['recall@10']:.4f} precision={row['precision@10']:.4f} "
            f"users={row['num_users_evaluated']} runtime={row['runtime_seconds']:.2f}s"
        )


def run_grid_search(config: Config) -> dict[str, Any]:
    print("Step A: Building/refreshing Item-CF matrix artifact...")
    matrix.run(config)

    print("Step B: Running baseline...")
    baseline_record = _run_single_experiment(config=config, label="baseline")
    _print_summary("Baseline", [baseline_record], top_n=1)

    print("Step C: Running Item-CF hyperparameter grid search...")
    grid = {
        "use_positive_only": [True, False],
        "top_k_neighbors": [25, 50, 100, 200, 400],
        "min_similarity": [0.0, 0.01, 0.05, 0.1, 0.2],
        "normalize_scores": [True, False],
    }
    keys = list(grid.keys())

    records: list[dict[str, Any]] = [baseline_record]
    combinations = list(itertools.product(*(grid[key] for key in keys)))
    total = len(combinations)
    for run_idx, values in enumerate(combinations, start=1):
        overrides = dict(zip(keys, values, strict=True))
        label = f"grid_{run_idx:03d}"
        print(f"[{run_idx}/{total}] {label} -> {overrides}")
        records.append(
            _run_single_experiment(
                config=config,
                label=label,
                overrides=overrides,
            )
        )

    ranked = _sort_records(records)
    _print_summary("Top Runs (ranked by ndcg, recall, precision)", ranked)

    summary = {
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "item_cf",
        "search_grid": grid,
        "baseline": baseline_record,
        "ranked_runs": ranked,
        "best_run": ranked[0] if ranked else None,
    }
    output_path: Path = (
        config.data_dirs.model_assets_dir / "item_cf_tuning_results.json"
    )
    with open(output_path, "w") as file_obj:
        json.dump(summary, file_obj, indent=2)
    print(f"\nSaved tuning summary to {output_path}")
    return summary


if __name__ == "__main__":
    run_grid_search(load_config())
