from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    ascii_words = re.findall(r"[a-z0-9_\.-]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    cjk_bigrams = ["".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))]
    return ascii_words + chinese + cjk_bigrams


class HashEmbedding:
    """Dependency-free deterministic embedding for tests/degraded local mode."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            for token in _tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            rows.append([x / norm for x in vec])
        return rows


class FastEmbedEmbedding:
    """Local ONNX embedding backend powered by FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "FastEmbed 未安装。请执行 `pip install -e '.[agent]'`，"
                "或把 TRADINGAGENTS_RAG_EMBEDDING_BACKEND 设置为 hash 仅用于本地测试。"
            ) from exc
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        probe = list(self._model.embed(["dimension probe"]))[0]
        self.dimension = int(len(probe))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, row)) for row in self._model.embed(list(texts))]


class FastEmbedReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("FastEmbed reranker 不可用") from exc
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(x) for x in self._model.rerank(query, list(documents))]


def build_embedder(config: dict):
    backend = str(config.get("rag_embedding_backend", "fastembed")).strip().lower()
    if backend == "hash":
        return HashEmbedding(int(config.get("rag_hash_dimension", 256)))
    if backend == "fastembed":
        return FastEmbedEmbedding(str(config.get("rag_embedding_model", "BAAI/bge-small-zh-v1.5")))
    raise ValueError(f"不支持的 RAG embedding backend: {backend}")


def build_reranker(config: dict):
    if not config.get("rag_reranker_enabled", True):
        return None
    try:
        return FastEmbedReranker(str(config.get("rag_reranker_model", "BAAI/bge-reranker-base")))
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG reranker 不可用，将使用 Hybrid/RRF 排名: %s", exc)
        return None
