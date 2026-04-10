"""Unit tests for rag.drift.snapshot."""

from __future__ import annotations

import numpy as np
import pytest

from rag.drift.snapshot import DistributionSnapshot
from rag.models import DriftConfig, DriftResult

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DIM = 16
N_REF = 200
N_QUERY = 80
RNG_SEED = 42


def _cfg(**kwargs: object) -> DriftConfig:
    defaults: dict[str, object] = {"pca_components": 8, "threshold_alpha": 0.2}
    defaults.update(kwargs)
    return DriftConfig(**defaults)  # type: ignore[arg-type]


def _unit_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalise rows of *arr*."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.asarray(arr / np.where(norms == 0, 1.0, norms), dtype=np.float32)


def _same_dist_vecs(n: int, seed: int = RNG_SEED) -> np.ndarray:
    """Return *n* unit-norm vectors drawn from the reference distribution."""
    rng = np.random.default_rng(seed)
    return _unit_rows(rng.standard_normal((n, DIM)).astype(np.float32))


def _shifted_dist_vecs(n: int, shift: float = 5.0, seed: int = RNG_SEED + 1) -> np.ndarray:
    """Return *n* unit-norm vectors from a clearly shifted distribution."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, DIM)).astype(np.float32) + shift
    return _unit_rows(raw)


@pytest.fixture
def snapshot() -> DistributionSnapshot:
    ref = _same_dist_vecs(N_REF)
    return DistributionSnapshot(ref, _cfg())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_builds_without_error(self) -> None:
        ref = _same_dist_vecs(N_REF)
        DistributionSnapshot(ref, _cfg())  # must not raise

    def test_snapshot_size_property(self) -> None:
        ref = _same_dist_vecs(N_REF)
        snap = DistributionSnapshot(ref, _cfg())
        assert snap.snapshot_size == N_REF

    def test_n_components_capped_at_dim(self) -> None:
        """pca_components > dim should be silently capped."""
        ref = _same_dist_vecs(50)
        snap = DistributionSnapshot(ref, _cfg(pca_components=1000))
        assert snap.n_components <= DIM

    def test_n_components_capped_at_n_minus_1(self) -> None:
        """pca_components must not exceed n-1 (SVD constraint)."""
        ref = _same_dist_vecs(10)
        snap = DistributionSnapshot(ref, _cfg(pca_components=100))
        assert snap.n_components <= 9

    def test_requires_at_least_two_embeddings(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            DistributionSnapshot(_same_dist_vecs(1), _cfg())

    def test_empty_embeddings_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            DistributionSnapshot(np.empty((0, DIM), dtype=np.float32), _cfg())


# ---------------------------------------------------------------------------
# compare — return type and structure
# ---------------------------------------------------------------------------


class TestCompareReturnType:
    def test_returns_drift_result(self, snapshot: DistributionSnapshot) -> None:
        result = snapshot.compare(_same_dist_vecs(N_QUERY))
        assert isinstance(result, DriftResult)

    def test_statistic_in_unit_interval(self, snapshot: DistributionSnapshot) -> None:
        result = snapshot.compare(_same_dist_vecs(N_QUERY))
        assert 0.0 <= result.statistic <= 1.0

    def test_pvalue_in_unit_interval(self, snapshot: DistributionSnapshot) -> None:
        result = snapshot.compare(_same_dist_vecs(N_QUERY))
        assert 0.0 <= result.pvalue <= 1.0

    def test_window_size_matches_query_count(self, snapshot: DistributionSnapshot) -> None:
        queries = _same_dist_vecs(37)
        result = snapshot.compare(queries)
        assert result.window_size == 37

    def test_snapshot_size_in_result(self, snapshot: DistributionSnapshot) -> None:
        result = snapshot.compare(_same_dist_vecs(N_QUERY))
        assert result.snapshot_size == N_REF

    def test_empty_query_raises(self, snapshot: DistributionSnapshot) -> None:
        with pytest.raises(ValueError, match="empty"):
            snapshot.compare(np.empty((0, DIM), dtype=np.float32))

    def test_wrong_dim_raises(self, snapshot: DistributionSnapshot) -> None:
        wrong_dim = _same_dist_vecs(10)[:, :8]  # DIM//2 dimensions
        with pytest.raises(ValueError, match="dim"):
            snapshot.compare(wrong_dim)


# ---------------------------------------------------------------------------
# compare — statistical correctness (same distribution → low, shifted → high)
# ---------------------------------------------------------------------------


class TestCompareDistributions:
    def test_same_distribution_low_statistic(self, snapshot: DistributionSnapshot) -> None:
        """Queries from the same distribution should produce a low KS statistic."""
        result = snapshot.compare(_same_dist_vecs(N_QUERY, seed=99))
        # KS stat should be well below 0.5 for well-matched distributions
        assert result.statistic < 0.5

    def test_shifted_distribution_high_statistic(self, snapshot: DistributionSnapshot) -> None:
        """A large shift should produce a high KS statistic."""
        result = snapshot.compare(_shifted_dist_vecs(N_QUERY, shift=10.0))
        assert result.statistic > 0.5

    def test_same_dist_not_flagged_as_drifted(self, snapshot: DistributionSnapshot) -> None:
        """No drift alert for in-distribution queries."""
        result = snapshot.compare(_same_dist_vecs(N_QUERY, seed=7))
        assert not result.drifted

    def test_shifted_dist_flagged_as_drifted(self, snapshot: DistributionSnapshot) -> None:
        """Hard drift should be flagged when statistic > threshold_alpha."""
        snap = DistributionSnapshot(_same_dist_vecs(N_REF), _cfg(threshold_alpha=0.05))
        result = snap.compare(_shifted_dist_vecs(N_QUERY, shift=10.0))
        assert result.drifted

    def test_statistic_monotone_with_shift(self, snapshot: DistributionSnapshot) -> None:
        """Larger distribution shift should produce larger KS statistic."""
        stat_small = snapshot.compare(_shifted_dist_vecs(N_QUERY, shift=1.0)).statistic
        stat_large = snapshot.compare(_shifted_dist_vecs(N_QUERY, shift=8.0)).statistic
        assert stat_large > stat_small

    def test_drifted_flag_consistent_with_threshold(self) -> None:
        """drifted == (statistic > threshold_alpha)."""
        ref = _same_dist_vecs(N_REF)
        for alpha in (0.05, 0.3, 0.7):
            snap = DistributionSnapshot(ref, _cfg(threshold_alpha=alpha))
            result = snap.compare(_shifted_dist_vecs(N_QUERY, shift=5.0))
            assert result.drifted == (result.statistic > alpha)

    def test_single_query_vector_accepted(self, snapshot: DistributionSnapshot) -> None:
        """A window of size 1 is valid (edge case)."""
        result = snapshot.compare(_same_dist_vecs(1))
        assert isinstance(result, DriftResult)
