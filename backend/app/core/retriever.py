"""混合检索（Day 11）。

**为什么必须混合？** 两路检索的失败模式正好互补：

| 场景 | 纯向量 | 纯 BM25 | 混合 |
|---|---|---|---|
| 问「MCP 是什么」，文档里写着 "MCP" | 可能被"协议"泛化段落挤掉 | ✅ 精准命中 | ✅ |
| 问「怎么让模型不瞎编」，文档写的是"抑制幻觉" | ✅ 语义命中 | ❌ 字面不重合，漏 | ✅ |
| 问某个 API 名 / 报错码 | ❌ 专有名词弱 | ✅ | ✅ |

**融合为什么用 RRF 而不是加权求和？**
因为两路的分数量纲完全不同：余弦相似度在 0~1，BM25 是无上界的实数。
直接加权求和，BM25 那一路会一边倒地压过向量分。
RRF（Reciprocal Rank Fusion）只看**排名**不看分数：
    score = Σ 1 / (k + rank)，k 通常取 60
这样天然免疫量纲问题，也不需要调权重。这是面试的高分答案。

（加权求和也不是不能用，但必须先各自归一化 —— 配置里留了 `hybrid_fusion` 开关。）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.core.bm25 import BM25
from app.core.embeddings import EmbeddingProvider
from app.core.vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "fused_score": round(self.fused_score, 4),
        }


class HybridRetriever:
    """向量 + BM25 混合检索器。

    BM25 索引是**懒构建**的：第一次检索时才在全部 chunk 上 fit。
    原因：BM25 是全局统计量（IDF 依赖全语料），索引变了就必须重建。
    这也是为什么 Day 11 要建评测集 —— 改了切块参数，BM25 得整体重算。
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings or get_settings()
        self._bm25: BM25 | None = None
        self._bm25_corpus_size = -1

    # ---------- 索引 ----------

    def invalidate(self) -> None:
        """索引变更后必须调用，否则 BM25 用的是过期语料。"""
        self._bm25 = None
        self._bm25_corpus_size = -1

    def _ensure_bm25(self) -> BM25 | None:
        n = self.store.count()
        if n == 0:
            return None
        if self._bm25 is None or self._bm25_corpus_size != n:
            logger.info("构建 BM25 索引（%d 条）", n)
            self._bm25 = BM25().fit(self.store.texts())
            self._bm25_corpus_size = n
        return self._bm25

    # ---------- 检索 ----------

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> list[RetrievedChunk]:
        mode = (mode or self.settings.retrieval_mode).lower()
        top_k = top_k or self.settings.rag_top_k

        if self.store.count() == 0:
            return []

        if mode == "vector":
            return await self._vector_only(query, top_k)
        if mode == "bm25":
            return self._bm25_only(query, top_k)
        if mode == "hybrid":
            return await self._hybrid(query, top_k)

        raise ValueError(f"未知检索模式 `{mode}`，可选：vector / bm25 / hybrid")

    # ---------- 三路实现 ----------

    async def _vector_only(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vecs = await self.embedder.embed([query])
        hits = self.store.search(vecs[0], top_k)
        return [
            RetrievedChunk(
                text=rec.text,
                source=rec.source,
                page=rec.page,
                vector_score=score,
                fused_score=score,
            )
            for rec, score in hits
        ]

    def _bm25_only(self, query: str, top_k: int) -> list[RetrievedChunk]:
        bm25 = self._ensure_bm25()
        if bm25 is None:
            return []
        scores = bm25.score(query)
        records = self.store.records()
        ranked = sorted(
            zip(records, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            RetrievedChunk(
                text=rec.text,
                source=rec.source,
                page=rec.page,
                bm25_score=score,
                fused_score=score,
            )
            for rec, score in ranked
        ]

    async def _hybrid(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """粗排取 fetch_k 条，再用 RRF 融合取 top_k。"""
        fetch_k = max(self.settings.rag_fetch_k, top_k)

        vecs = await self.embedder.embed([query])
        vector_hits = self.store.search(vecs[0], fetch_k)

        bm25 = self._ensure_bm25()
        records = self.store.records()
        bm25_scores = bm25.score(query) if bm25 else [0.0] * len(records)
        bm25_hits = sorted(
            zip(records, bm25_scores), key=lambda x: x[1], reverse=True
        )[:fetch_k]

        # 用 (source, index) 做唯一键，避免相同文本互相覆盖
        merged: dict[tuple, RetrievedChunk] = {}

        def key_of(rec: VectorRecord) -> tuple:
            return (rec.source, rec.metadata.get("index"), rec.text[:32])

        for rank, (rec, score) in enumerate(vector_hits):
            k = key_of(rec)
            item = merged.setdefault(
                k, RetrievedChunk(text=rec.text, source=rec.source, page=rec.page)
            )
            item.vector_score = score
            item.fused_score += 1.0 / (RRF_K + rank + 1)

        for rank, (rec, score) in enumerate(bm25_hits):
            k = key_of(rec)
            item = merged.setdefault(
                k, RetrievedChunk(text=rec.text, source=rec.source, page=rec.page)
            )
            item.bm25_score = score
            item.fused_score += 1.0 / (RRF_K + rank + 1)

        out = sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)[:top_k]

        # 把 RRF 分归一化到 0~1，纯粹为了前端显示好看
        if out:
            top = out[0].fused_score or 1.0
            for c in out:
                c.fused_score = c.fused_score / top

        return out
