"""Index distribution snapshot: captures mean/covariance of chunk embeddings at build time."""

from __future__ import annotations

import numpy as np
from scipy import stats

from rag.models import DriftConfig, DriftResult


class DistributionSnapshot:
    """Captures the reference embedding distribution at index-build time and
    compares incoming query batches against it using a two-sample KS test.

    Algorithm
    ---------
    1. **Fit** (``__init__``): centre the chunk embeddings, compute a truncated
       SVD to obtain the top-*k* PCA components, project the reference set to
       *k* dimensions, and store the projected values column-wise as the
       reference distribution.
    2. **Compare**: project the query batch onto the *same* PCA basis (using
       the reference mean and components — no re-fitting), then run a
       two-sample KS test independently for each PCA dimension.  Return the
       *maximum* KS statistic across all dimensions together with its p-value.

    The max-statistic aggregation is deliberately conservative: it fires if
    *any* latent dimension drifts, which is what matters for retrieval quality.

    Args:
        embeddings: Float32 array of shape ``(n_chunks, dim)`` — the full set
            of chunk embeddings stored in the vector store at index-build time.
        config: Drift detection configuration (``pca_components``,
            ``threshold_alpha``, …).

    Raises:
        ValueError: If ``embeddings`` has fewer than 2 rows, or if
            ``pca_components`` exceeds the embedding dimensionality.
    """

    def __init__(self, embeddings: np.ndarray, config: DriftConfig) -> None:
        if embeddings.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 embeddings to build a snapshot, got {embeddings.shape[0]}."
            )
        n = int(embeddings.shape[0])
        dim = int(embeddings.shape[1])
        n_components = min(int(config.pca_components), dim, n - 1)

        # --- PCA via truncated SVD (no sklearn dependency) ---
        self._ref_mean: np.ndarray = np.asarray(embeddings.mean(axis=0), dtype=np.float32)
        centered: np.ndarray = np.asarray(embeddings - self._ref_mean, dtype=np.float32)

        # SVD: U (n×k), s (k,), Vt (k×dim).  Vt rows are the principal axes.
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self._components: np.ndarray = np.asarray(Vt[:n_components], dtype=np.float32)

        # Project reference embeddings: (n, n_components)
        self._ref_projected: np.ndarray = np.asarray(
            centered @ self._components.T, dtype=np.float32
        )

        self._config = config
        self._n_ref = n

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of PCA dimensions actually used (≤ config.pca_components)."""
        return self._components.shape[0]  # type: ignore[no-any-return]  # numpy shape → Any

    @property
    def snapshot_size(self) -> int:
        """Number of chunk embeddings in the reference distribution."""
        return self._n_ref  # stored as plain int in __init__

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        """Project *embeddings* onto the stored PCA basis.

        Args:
            embeddings: Float32 array of shape ``(n, dim)``.

        Returns:
            Float32 array of shape ``(n, n_components)``.
        """
        centered: np.ndarray = np.asarray(embeddings - self._ref_mean, dtype=np.float32)
        return np.asarray(centered @ self._components.T, dtype=np.float32)

    def compare(self, query_embeddings: np.ndarray) -> DriftResult:
        """Measure distributional shift between the snapshot and *query_embeddings*.

        Projects *query_embeddings* onto the reference PCA basis, then runs an
        independent two-sample KS test for each dimension.  Returns the maximum
        KS statistic (most-drifted dimension) with its associated p-value.

        Args:
            query_embeddings: Float32 array of shape ``(n_queries, dim)``.
                Must have at least 1 row and the same ``dim`` as the snapshot.

        Returns:
            :class:`~rag.models.DriftResult` with ``statistic``, ``pvalue``,
            ``drifted``, ``window_size``, and ``snapshot_size`` populated.

        Raises:
            ValueError: If ``query_embeddings`` is empty or has wrong ``dim``.
        """
        if query_embeddings.shape[0] == 0:
            raise ValueError("query_embeddings must not be empty.")
        if query_embeddings.shape[1] != self._ref_mean.shape[0]:
            raise ValueError(
                f"query_embeddings dim {query_embeddings.shape[1]} does not match "
                f"snapshot dim {self._ref_mean.shape[0]}."
            )

        query_proj: np.ndarray = self._project(query_embeddings)

        max_stat = 0.0
        pvalue_at_max = 1.0

        for dim_i in range(self.n_components):
            ref_col: np.ndarray = np.asarray(self._ref_projected[:, dim_i])
            qry_col: np.ndarray = np.asarray(query_proj[:, dim_i])
            result = stats.ks_2samp(ref_col, qry_col)
            stat = float(result.statistic)
            if stat > max_stat:
                max_stat = stat
                pvalue_at_max = float(result.pvalue)

        return DriftResult(
            statistic=max_stat,
            pvalue=pvalue_at_max,
            drifted=max_stat > self._config.threshold_alpha,
            window_size=int(query_embeddings.shape[0]),
            snapshot_size=self._n_ref,
        )
