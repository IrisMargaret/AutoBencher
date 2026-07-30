"""Text-Dedup, datasketch, and Sentence-Transformers similarity backends.

The deterministic MinHash design is a clean-room adaptation of the algorithm
used by ChenghaoMou/text-dedup (Apache-2.0); no source file is copied.  When
available, ekzhu/datasketch supplies the production MinHash and LSH index.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class SemanticSimilarityUnavailable(RuntimeError):
    """Raised when a required Sentence-Transformers backend cannot load."""


class MinHashBackendUnavailable(RuntimeError):
    """Raised when the required datasketch backend cannot load."""


def _similarity_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _word_shingles(value: Any, ngram_size: int) -> set[str]:
    tokens = re.findall(
        r"[a-z0-9_]+|[^\w\s]",
        _similarity_text(value),
        flags=re.UNICODE,
    )
    if not tokens:
        return set()
    width = max(1, int(ngram_size))
    if len(tokens) < width:
        return {"\u241f".join(tokens)}
    return {
        "\u241f".join(tokens[index:index + width])
        for index in range(len(tokens) - width + 1)
    }


def _fallback_minhash_signature(
    value: Any,
    *,
    num_perm: int,
    ngram_size: int,
    seed: int,
) -> tuple[int, ...]:
    shingles = _word_shingles(value, ngram_size)
    if not shingles:
        return tuple(0 for _ in range(int(num_perm)))
    signature = []
    for permutation in range(int(num_perm)):
        person = (
            f"ab{int(seed) & 0xFFFF:04x}{permutation & 0xFFFF:04x}"
        ).encode("ascii")
        signature.append(
            min(
                int.from_bytes(
                    hashlib.blake2b(
                        shingle.encode("utf-8"),
                        digest_size=8,
                        person=person,
                    ).digest(),
                    "big",
                )
                for shingle in shingles
            )
        )
    return tuple(signature)


def _datasketch_minhash(
    value: Any,
    *,
    num_perm: int,
    ngram_size: int,
    seed: int,
) -> tuple[tuple[int, ...], Any]:
    try:
        from datasketch import MinHash
    except ImportError as exc:
        raise MinHashBackendUnavailable("datasketch is not installed") from exc
    try:
        sketch = MinHash(num_perm=int(num_perm), seed=int(seed))
        shingles = sorted(_word_shingles(value, ngram_size))
        if shingles:
            update_batch = getattr(sketch, "update_batch", None)
            encoded = [item.encode("utf-8") for item in shingles]
            if callable(update_batch):
                update_batch(encoded)
            else:
                for item in encoded:
                    sketch.update(item)
        signature = tuple(int(value) for value in sketch.hashvalues)
    except Exception as exc:
        raise MinHashBackendUnavailable(
            f"datasketch MinHash initialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return signature, sketch


def minhash_signature(
    value: Any,
    *,
    num_perm: int,
    ngram_size: int,
    seed: int,
    use_datasketch: bool = True,
    require_datasketch: bool = False,
) -> tuple[int, ...]:
    """Return a deterministic MinHash signature with a controlled fallback."""
    if use_datasketch:
        try:
            signature, _ = _datasketch_minhash(
                value,
                num_perm=num_perm,
                ngram_size=ngram_size,
                seed=seed,
            )
            return signature
        except MinHashBackendUnavailable:
            if require_datasketch:
                raise
    return _fallback_minhash_signature(
        value,
        num_perm=num_perm,
        ngram_size=ngram_size,
        seed=seed,
    )


def minhash_similarity(
    left_signature: Sequence[int],
    right_signature: Sequence[int],
) -> float:
    if len(left_signature) != len(right_signature):
        raise ValueError("MinHash signatures must have equal length")
    if not left_signature:
        return 1.0
    return sum(
        left == right
        for left, right in zip(left_signature, right_signature)
    ) / len(left_signature)


_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _load_sentence_transformer(config: Mapping[str, Any]) -> Any:
    key = (
        str(config["sentence_transformers_model"]),
        str(config.get("sentence_transformers_device") or "cpu"),
        str(config.get("sentence_transformers_cache_dir") or ""),
        bool(config["sentence_transformers_local_files_only"]),
        str(config.get("sentence_transformers_revision") or ""),
    )
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SemanticSimilarityUnavailable(
            "sentence-transformers is not installed"
        ) from exc
    kwargs = {
        "device": key[1],
        "cache_folder": key[2] or None,
        "local_files_only": key[3],
        "trust_remote_code": False,
    }
    if key[4]:
        kwargs["revision"] = key[4]
    try:
        model = SentenceTransformer(key[0], **kwargs)
    except Exception as exc:
        raise SemanticSimilarityUnavailable(
            f"unable to load Sentence-Transformers model {key[0]!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _MODEL_CACHE[key] = model
    return model


@dataclass(frozen=True)
class SimilarityBatch:
    """Precomputed lexical and semantic representations for one text list."""

    texts: tuple[str, ...]
    minhash_signatures: tuple[tuple[int, ...], ...]
    embeddings: np.ndarray | None
    minhash_backend: str
    minhash_error: str | None
    minhash_lsh: Any | None
    datasketch_sketches: tuple[Any, ...]
    semantic_backend: str
    semantic_error: str | None

    def minhash_candidates(self, index: int) -> tuple[int, ...]:
        """Return LSH candidates, while callers retain exact fallback checks."""
        if self.minhash_lsh is None or not self.datasketch_sketches:
            return ()
        try:
            candidates = self.minhash_lsh.query(
                self.datasketch_sketches[index]
            )
        except Exception:
            return ()
        return tuple(
            sorted(
                int(candidate)
                for candidate in candidates
                if int(candidate) != int(index)
            )
        )

    def pair(self, left_index: int, right_index: int) -> dict[str, Any]:
        minhash_score = minhash_similarity(
            self.minhash_signatures[left_index],
            self.minhash_signatures[right_index],
        )
        semantic_score = None
        if self.embeddings is not None:
            semantic_score = float(
                np.clip(
                    np.dot(
                        self.embeddings[left_index],
                        self.embeddings[right_index],
                    ),
                    -1.0,
                    1.0,
                )
            )
        return {
            "minhash_similarity": minhash_score,
            "minhash_backend": self.minhash_backend,
            "minhash_backend_error": self.minhash_error,
            "sentence_transformers_similarity": semantic_score,
            "semantic_backend": self.semantic_backend,
            "semantic_backend_error": self.semantic_error,
        }


def build_similarity_batch(
    texts: Sequence[Any],
    config: Mapping[str, Any],
) -> SimilarityBatch:
    """Precompute datasketch MinHash/LSH and Sentence embeddings once."""
    normalized_texts = tuple(_similarity_text(text) for text in texts)
    num_perm = int(config["text_dedup_num_perm"])
    ngram_size = int(config["text_dedup_ngram_size"])
    seed = int(config["text_dedup_seed"])
    minhash_enabled = bool(config.get("text_dedup_enabled", True))
    datasketch_enabled = bool(config.get("datasketch_enabled", True))
    datasketch_required = bool(config.get("datasketch_required", False))
    minhash_backend = "disabled"
    minhash_error = None
    datasketch_sketches: tuple[Any, ...] = ()
    minhash_lsh = None
    if minhash_enabled and datasketch_enabled and normalized_texts:
        try:
            generated = tuple(
                _datasketch_minhash(
                    text,
                    num_perm=num_perm,
                    ngram_size=ngram_size,
                    seed=seed,
                )
                for text in normalized_texts
            )
            signatures = tuple(item[0] for item in generated)
            datasketch_sketches = tuple(item[1] for item in generated)
            minhash_backend = "datasketch"
            if bool(config.get("datasketch_lsh_enabled", True)):
                from datasketch import MinHashLSH

                lsh_threshold = min(
                    float(config["text_dedup_similarity_threshold"]),
                    float(
                        config["holdout_text_dedup_similarity_threshold"]
                    ),
                )
                minhash_lsh = MinHashLSH(
                    threshold=lsh_threshold,
                    num_perm=num_perm,
                )
                for index, sketch in enumerate(datasketch_sketches):
                    minhash_lsh.insert(str(index), sketch)
        except (ImportError, MinHashBackendUnavailable, ValueError) as exc:
            minhash_error = f"{type(exc).__name__}: {exc}"
            if datasketch_required:
                raise MinHashBackendUnavailable(minhash_error) from exc
            signatures = tuple(
                _fallback_minhash_signature(
                    text,
                    num_perm=num_perm,
                    ngram_size=ngram_size,
                    seed=seed,
                )
                for text in normalized_texts
            )
            datasketch_sketches = ()
            minhash_lsh = None
            minhash_backend = "builtin_minhash_exhaustive"
    elif minhash_enabled:
        signatures = tuple(
            _fallback_minhash_signature(
                text,
                num_perm=num_perm,
                ngram_size=ngram_size,
                seed=seed,
            )
            for text in normalized_texts
        )
        minhash_backend = "builtin_minhash_exhaustive"
    else:
        signatures = tuple(
            tuple(0 for _ in range(num_perm))
            for _ in normalized_texts
        )
    embeddings = None
    backend = "disabled"
    backend_error = None
    if bool(config["sentence_transformers_enabled"]) and normalized_texts:
        try:
            model = _load_sentence_transformer(config)
            encoded = model.encode(
                list(normalized_texts),
                batch_size=int(config["sentence_transformers_batch_size"]),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(encoded, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(
                normalized_texts
            ):
                raise SemanticSimilarityUnavailable(
                    "Sentence-Transformers returned an invalid embedding shape"
                )
            backend = "sentence_transformers"
        except SemanticSimilarityUnavailable as exc:
            backend = "unavailable"
            backend_error = str(exc)
        except Exception as exc:
            backend = "unavailable"
            backend_error = f"{type(exc).__name__}: {exc}"
    if (
        bool(config["sentence_transformers_required"])
        and normalized_texts
        and backend != "sentence_transformers"
    ):
        raise SemanticSimilarityUnavailable(
            backend_error
            or "required Sentence-Transformers backend is disabled"
        )
    return SimilarityBatch(
        texts=normalized_texts,
        minhash_signatures=signatures,
        embeddings=embeddings,
        minhash_backend=minhash_backend,
        minhash_error=minhash_error,
        minhash_lsh=minhash_lsh,
        datasketch_sketches=datasketch_sketches,
        semantic_backend=backend,
        semantic_error=backend_error,
    )
