# State

## Current Position
- Planning Phase 1: ML Pipeline Testing
- Branch: ml-pipeline-testing

## Existing Test Coverage (122 tests passing)
### Unit Tests (8 files, ~48 tests):
- test_preprocess_movies.py - extract_year, clean_title, split_genres
- test_preprocess_ratings.py - map_rating_to_bucket, bucket_to_preference
- test_filtering.py - iterative_core_filter
- test_split.py - chronological_split
- test_evaluate.py - dcg_at_k
- test_online_updater.py - update_user_vector
- test_artifact_loader.py - _ensure_artifact_paths_exist
- test_recommender_utils.py - _to_int_user_id, _top_n_indices

### Integration Tests (3 files, ~74 tests):
- test_full_pipeline.py - end-to-end offline→online pipeline
- test_online_recommender.py - online recommender with synthetic data
- test_online_recommender_debug.py - verbose debug duplicate (cleanup candidate)

### Tests in src/ (not in tests/ directory):
- test_feedback_mapping.py - swipe_to_preference (should be consolidated)

## Identified Gaps
1. No unit tests for build_matrix.py (sparse matrix construction, _to_native)
2. No unit tests for Recommender class internals (_require_artifacts, _cold_start_vector, _base_user_vector, _current_user_vector)
3. feedback_mapping tests exist in src/ but not in tests/unit/
4. No API endpoint tests (FastAPI TestClient)
5. No FeedManager service tests
6. No MovieHydrator service tests
7. test_online_recommender_debug.py is redundant duplicate
