"""Unit tests for rag.embedding.encoder."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.embedding.encoder import Encoder, SentenceTransformerEncoder, _l2_normalize

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_st(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock, None, None]:
    """Patch SentenceTransformer so no model is downloaded during tests."""
    with patch("rag.embedding.encoder.SentenceTransformer") as MockST:
        instance = MockST.return_value
        # Default: return two unnormalised vectors
        instance.encode.return_value = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        yield instance


def _make_encoder(mock_st: MagicMock, batch_size: int = 64) -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder("mock-model", batch_size=batch_size)


# ---------------------------------------------------------------------------
# _l2_normalize (pure function — no mock needed)
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_norm_rows(self) -> None:
        arr = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        out = _l2_normalize(arr)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_known_values(self) -> None:
        arr = np.array([[3.0, 4.0]], dtype=np.float32)
        out = _l2_normalize(arr)
        np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)

    def test_zero_vector_stays_zero(self) -> None:
        """A zero vector must not produce NaN — norm is treated as 1."""
        arr = np.array([[0.0, 0.0]], dtype=np.float32)
        out = _l2_normalize(arr)
        assert not np.any(np.isnan(out))
        np.testing.assert_array_equal(out, [[0.0, 0.0]])

    def test_already_normalised_unchanged(self) -> None:
        arr = np.array([[0.6, 0.8]], dtype=np.float32)
        out = _l2_normalize(arr)
        np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_dtype_preserved(self) -> None:
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        out = _l2_normalize(arr)
        assert out.dtype == np.float32

    def test_does_not_mutate_input(self) -> None:
        arr = np.array([[3.0, 4.0]], dtype=np.float32)
        original = arr.copy()
        _l2_normalize(arr)
        np.testing.assert_array_equal(arr, original)


# ---------------------------------------------------------------------------
# Encoder ABC
# ---------------------------------------------------------------------------


class TestEncoderABC:
    def test_is_abstract(self) -> None:
        """Encoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Encoder()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_encode(self) -> None:
        class Incomplete(Encoder):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_is_valid(self) -> None:
        class Minimal(Encoder):
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.zeros((len(texts), 4), dtype=np.float32)

        enc = Minimal()
        out = enc.encode(["hello"])
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# SentenceTransformerEncoder
# ---------------------------------------------------------------------------


class TestSentenceTransformerEncoder:
    def test_is_encoder_subclass(self, mock_st: MagicMock) -> None:
        enc = _make_encoder(mock_st)
        assert isinstance(enc, Encoder)

    def test_encode_returns_ndarray(self, mock_st: MagicMock) -> None:
        enc = _make_encoder(mock_st)
        result = enc.encode(["hello", "world"])
        assert isinstance(result, np.ndarray)

    def test_encode_shape(self, mock_st: MagicMock) -> None:
        """Output shape is (n_texts, embedding_dim)."""
        enc = _make_encoder(mock_st)
        result = enc.encode(["a", "b"])
        assert result.shape == (2, 2)

    def test_encode_l2_normalises(self, mock_st: MagicMock) -> None:
        """Every row of the output must have unit L2 norm."""
        enc = _make_encoder(mock_st)
        result = enc.encode(["hello", "world"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-6)

    def test_encode_passes_batch_size(self, mock_st: MagicMock) -> None:
        """Configured batch_size is forwarded to the underlying model."""
        enc = _make_encoder(mock_st, batch_size=16)
        enc.encode(["a", "b"])
        _, kwargs = mock_st.encode.call_args
        assert kwargs["batch_size"] == 16

    def test_encode_disables_progress_bar(self, mock_st: MagicMock) -> None:
        enc = _make_encoder(mock_st)
        enc.encode(["text"])
        _, kwargs = mock_st.encode.call_args
        assert kwargs["show_progress_bar"] is False

    def test_encode_requests_numpy_output(self, mock_st: MagicMock) -> None:
        enc = _make_encoder(mock_st)
        enc.encode(["text"])
        _, kwargs = mock_st.encode.call_args
        assert kwargs["convert_to_numpy"] is True

    def test_encode_does_not_delegate_normalisation(self, mock_st: MagicMock) -> None:
        """sentence-transformers normalisation is disabled; we own the step."""
        enc = _make_encoder(mock_st)
        enc.encode(["text"])
        _, kwargs = mock_st.encode.call_args
        assert kwargs["normalize_embeddings"] is False

    def test_encode_output_is_float32(self, mock_st: MagicMock) -> None:
        enc = _make_encoder(mock_st)
        result = enc.encode(["text"])
        assert result.dtype == np.float32

    def test_encode_zero_vector_input_no_nan(self, mock_st: MagicMock) -> None:
        """A zero embedding from the model must not produce NaN in the output."""
        mock_st.encode.return_value = np.array([[0.0, 0.0]], dtype=np.float32)
        enc = _make_encoder(mock_st)
        result = enc.encode(["empty"])
        assert not np.any(np.isnan(result))

    def test_sentence_transformer_instantiated_with_model_name(self) -> None:
        with patch("rag.embedding.encoder.SentenceTransformer") as MockST:
            MockST.return_value.encode.return_value = np.array([[1.0, 0.0]])
            SentenceTransformerEncoder("my-model")
            MockST.assert_called_once_with("my-model")
