"""向量库（Day 10）。

两个后端，同一套接口：

| backend | 实现 | 适用 |
|---|---|---|
| `memory` | 纯 Python 暴力检索 | 开发、演示、几千条以内 |
| `chroma` | ChromaDB 持久化 + ANN 索引 | 数据量上来了、需要重启不丢 |

**为什么默认用内存版？**
因为它的零依赖能让你 5 分钟跑通全流程，不被环境问题卡住。
而把两套实现放在同一个接口后面，本身就是面试素材：
「我做了抽象层，开发期用内存实现零依赖，数据量上来换 Chroma 只改一个配置。」

暴力检索的复杂度是 O(N·d)。一万条 × 1024 维大约几十毫秒，
演示完全够用 —— 但**要能说出这个瓶颈在哪**，以及 ANN（HNSW/IVF）怎么解决。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.core.embeddings import cosine

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        p = self.metadata.get("page")
        return int(p) if p is not None else None


class VectorStore(Protocol):
    def add(self, records: list[VectorRecord]) -> None: ...
    def search(self, vector: list[float], top_k: int) -> list[tuple[VectorRecord, float]]: ...
    def texts(self) -> list[str]: ...
    def records(self) -> list[VectorRecord]: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...
    @property
    def backend(self) -> str: ...


# ============================================================================
# 内存实现
# ============================================================================

class InMemoryVectorStore:
    """暴力检索。零依赖，够用到几千条。"""

    backend = "memory"

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, records: list[VectorRecord]) -> None:
        self._records.extend(records)

    def search(self, vector: list[float], top_k: int) -> list[tuple[VectorRecord, float]]:
        scored = [(rec, cosine(vector, rec.vector)) for rec in self._records]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def texts(self) -> list[str]:
        return [r.text for r in self._records]

    def records(self) -> list[VectorRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)


# ============================================================================
# Chroma 实现
# ============================================================================

class ChromaVectorStore:
    """ChromaDB 持久化后端。需要 pip install chromadb。"""

    backend = "chroma"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 chromadb。执行 pip install chromadb，"
                "或把 VECTOR_BACKEND 改回 memory"
            ) from exc

        self._client = chromadb.PersistentClient(path=self.settings.vector_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.settings.vector_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._cache: dict[str, VectorRecord] = {}

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._collection.add(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            documents=[r.text for r in records],
            metadatas=[r.metadata for r in records],
        )
        for r in records:
            self._cache[r.id] = r

    def search(self, vector: list[float], top_k: int) -> list[tuple[VectorRecord, float]]:
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, max(1, self._collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[VectorRecord, float]] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        for i, rid in enumerate(ids):
            # cosine 距离 → 相似度
            score = 1.0 - float(dists[i]) if i < len(dists) else 0.0
            rec = VectorRecord(
                id=rid,
                text=docs[i],
                vector=[],
                metadata=dict(metas[i] or {}),
            )
            out.append((rec, score))
        return out

    def texts(self) -> list[str]:
        res = self._collection.get(include=["documents"])
        return list((res.get("documents") or []))

    def records(self) -> list[VectorRecord]:
        res = self._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [
            VectorRecord(id=ids[i], text=docs[i], vector=[], metadata=dict(metas[i] or {}))
            for i in range(len(ids))
        ]

    def clear(self) -> None:
        self._client.delete_collection(self.settings.vector_collection)
        self._collection = self._client.get_or_create_collection(
            name=self.settings.vector_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._cache.clear()

    def count(self) -> int:
        return self._collection.count()


# ============================================================================
# 工厂
# ============================================================================

def build_vector_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    backend = settings.vector_backend.lower()

    if backend == "chroma":
        try:
            return ChromaVectorStore(settings)
        except RuntimeError as exc:
            logger.warning("Chroma 不可用（%s），降级为内存后端", exc)
            return InMemoryVectorStore()
    return InMemoryVectorStore()
