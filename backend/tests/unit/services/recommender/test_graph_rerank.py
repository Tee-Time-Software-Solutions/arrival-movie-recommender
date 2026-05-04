"""Unit tests for graph_rerank: blend, shortlist, and Cypher param shape."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from movie_recommender.services.knowledge_graph.beacon import BeaconEntry
from movie_recommender.services.recommender.pipeline.online.serving.graph_rerank import (
    als_shortlist,
    blend_scores,
    compute_graph_scores,
)


class TestBlendScores:
    def test_weight_zero_is_noop(self):
        als = np.array([0.5, 0.3, 0.9, 0.1], dtype=np.float32)
        graph = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = blend_scores(als, graph, weight=0.0)
        np.testing.assert_array_equal(out, als)

    def test_empty_graph_is_noop(self):
        als = np.array([0.5, 0.3], dtype=np.float32)
        graph = np.array([], dtype=np.float32)
        out = blend_scores(als, graph, weight=0.5)
        np.testing.assert_array_equal(out, als)

    def test_zscore_balanced(self):
        # When weight=0.5 and the two normalized vectors are perfectly
        # anti-correlated, the blend collapses to ~zero. Verify that the
        # normalization actually happens.
        als = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        graph = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        out = blend_scores(als, graph, weight=0.5)
        # Output mean should be ~0 (z-score guarantees mean=0 per side, blend preserves it).
        assert abs(float(out.mean())) < 1e-5

    def test_negative_graph_penalizes_candidate(self):
        # Two candidates with the same ALS score; one has a strongly negative
        # graph score, the other strongly positive. After blending, the
        # negative-graph candidate must rank lower.
        als = np.array([0.5, 0.5, 0.4], dtype=np.float32)  # [tied, tied, lower]
        graph = np.array([10.0, -10.0, 0.0], dtype=np.float32)
        out = blend_scores(als, graph, weight=0.5)
        assert out[0] > out[1]


class TestAlsShortlist:
    def test_picks_top_k(self):
        scores = np.array([0.1, 0.9, 0.3, 0.7, 0.2], dtype=np.float32)
        ids = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        idx, picked_ids = als_shortlist(scores, ids, top_k=2)
        # The two highest are 0.9 (id=20) and 0.7 (id=40). argpartition does not
        # guarantee order within the top-K, so compare as sets.
        assert set(picked_ids.tolist()) == {20, 40}
        assert set(idx.tolist()) == {1, 3}

    def test_top_k_larger_than_pool_returns_all(self):
        scores = np.array([0.1, 0.9, 0.3], dtype=np.float32)
        ids = np.array([10, 20, 30], dtype=np.int32)
        idx, picked_ids = als_shortlist(scores, ids, top_k=10)
        assert idx.tolist() == [0, 1, 2]
        assert picked_ids.tolist() == [10, 20, 30]


class TestComputeGraphScores:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_candidates(self):
        driver = MagicMock()
        out = await compute_graph_scores(driver, [], beacon_map={})
        assert out == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_beacon(self):
        driver = MagicMock()
        out = await compute_graph_scores(driver, [1, 2, 3], beacon_map={})
        assert out == {}

    @pytest.mark.asyncio
    async def test_query_param_shape(self):
        # Mock the async session/result chain and assert we passed the right
        # params: stringified weight-map keys, integer id lists per entity type,
        # and a `candidates` list.
        beacon_map = {
            ("Director", 100): BeaconEntry("Director", 100, "Spielberg", 5.0),
            ("Actor", 200): BeaconEntry("Actor", 200, "Hanks", 3.0),
            ("Genre", 300): BeaconEntry("Genre", 300, "Drama", -2.0),
        }

        result_mock = MagicMock()
        result_mock.data = AsyncMock(
            return_value=[{"cand_tmdb_id": 42, "graph_score": 1.5}]
        )

        session_mock = MagicMock()
        session_mock.run = AsyncMock(return_value=result_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=None)

        driver = MagicMock()
        driver.session = MagicMock(return_value=session_mock)

        out = await compute_graph_scores(driver, [42], beacon_map)

        assert out == {42: 1.5}

        # session.run was called with kwargs; inspect them.
        kwargs = session_mock.run.call_args.kwargs
        assert kwargs["candidates"] == [42]
        assert kwargs["director_ids"] == [100]
        assert kwargs["actor_ids"] == [200]
        assert kwargs["writer_ids"] == []
        assert kwargs["genre_ids"] == [300]
        assert kwargs["keyword_ids"] == []
        # Cypher map keys must be strings.
        assert kwargs["director_w"] == {"100": 5.0}
        assert kwargs["actor_w"] == {"200": 3.0}
        assert kwargs["genre_w"] == {"300": -2.0}

    @pytest.mark.asyncio
    async def test_returns_empty_on_neo4j_error(self):
        driver = MagicMock()
        driver.session = MagicMock(side_effect=RuntimeError("neo4j down"))

        beacon_map = {
            ("Director", 100): BeaconEntry("Director", 100, "Spielberg", 5.0),
        }
        out = await compute_graph_scores(driver, [1], beacon_map)
        assert out == {}
