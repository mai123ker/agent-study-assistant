"""RAG 链路（Day 8-11 的组装）。

完整链路：

    文档目录
       │  loader.py        解析 → 文本 + 来源 + 页码
       ▼
    切块 splitter.py       按策略切 + overlap
       │
       ▼
    Embedding             向量化
       │
       ▼
    向量库 vector_store    入库
       │
    ┌──┴──────────────┐
    ▼                 ▼
  向量检索          BM25 检索      ← retriever.py
    └──────┬──────────┘
           ▼  RRF 融合
        top_k 片段
           │
           ▼  拼进 prompt（带约束 + 编号）
        模型生成
           │
           ▼
      答案 + 引用来源

**防幻觉的三道闸**（面试必答，就在 `_build_prompt` 里）：
1. 限定回答范围：「只能依据资料回答」
2. 允许承认不知道：「没有就说没有，不要编」
3. 要求给出处：引用处标 [1][2] —— 这既是给用户核查，也是逼模型别乱说

光靠 prompt 约束不够，第 4 周还要加评测（拒答类问题）来量化它到底管不管用。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.core import tools
from app.core.embeddings import EmbeddingProvider, build_embedding
from app.core.llm_client import LLMClient
from app.core.loader import load_dir
from app.core.retriever import HybridRetriever, RetrievedChunk
from app.core.splitter import split_document
from app.core.vector_store import VectorRecord, VectorStore, build_vector_store

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是资料助手，只能依据下面提供的资料回答用户问题。

必须遵守的规则：
1. 资料里没有的内容，直接回答「资料中没有相关内容」，绝对不要编造。
2. 引用资料时用 [1] [2] 这样的编号标注来源，编号对应资料前的序号。
3. 不要使用资料之外的任何知识，即使你知道答案。
4. 回答要简洁，直接给结论，再给依据。"""


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    used_context: bool = False


@dataclass
class IndexStats:
    indexed_files: int = 0
    total_chunks: int = 0
    strategy: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    elapsed_ms: int = 0
    files: list[str] = field(default_factory=list)


class RagService:
    """RAG 全链路服务。进程内单例（见 deps.py）。"""

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings | None = None,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.embedder = embedder or build_embedding(self.settings)
        self.store = store or build_vector_store(self.settings)
        self.retriever = HybridRetriever(self.store, self.embedder, self.settings)
        self._indexed = False
        self._stats = IndexStats(
            strategy=self.settings.chunk_strategy,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # ========================================================================
    # 索引
    # ========================================================================

    @property
    def is_indexed(self) -> bool:
        return self._indexed and self.store.count() > 0

    async def index(self, rebuild: bool = False, docs_dir: str | Path | None = None) -> IndexStats:
        start = time.perf_counter()

        if rebuild:
            self.store.clear()
            self.retriever.invalidate()
            self._indexed = False

        if self.is_indexed and not rebuild:
            logger.info("索引已存在（%d 条），跳过", self.store.count())
            return self._stats

        directory = Path(docs_dir) if docs_dir else self.settings.docs_path
        pages = load_dir(directory)

        if not pages:
            logger.warning("文档目录为空或不存在: %s", directory)
            self._stats = IndexStats(
                strategy=self.settings.chunk_strategy,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            self._indexed = True
            return self._stats

        # 切块
        chunks = []
        for page in pages:
            chunks.extend(
                split_document(
                    text=page.text,
                    source=page.source,
                    strategy=self.settings.chunk_strategy,
                    size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                    page=page.page,
                )
            )

        if not chunks:
            self._indexed = True
            return self._stats

        # 向量化（批量，减少 API 往返）
        vectors = await self.embedder.embed([c.text for c in chunks])

        records = [
            VectorRecord(
                id=f"{c.source}#{c.page or 0}#{c.index}",
                text=c.text,
                vector=vec,
                metadata={"source": c.source, "page": c.page, "index": c.index},
            )
            for c, vec in zip(chunks, vectors)
        ]
        self.store.add(records)
        self.retriever.invalidate()  # 语料变了，BM25 必须重建

        self._indexed = True
        self._stats = IndexStats(
            indexed_files=len({p.source for p in pages}),
            total_chunks=len(records),
            strategy=self.settings.chunk_strategy,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            files=sorted({p.source for p in pages}),
        )
        logger.info(
            "索引完成：%d 个文件 → %d 个片段，耗时 %dms",
            self._stats.indexed_files, self._stats.total_chunks, self._stats.elapsed_ms,
        )
        return self._stats

    def stats(self) -> dict:
        return {
            "indexed": self.is_indexed,
            "chunks": self.store.count(),
            "files": len({r.source for r in self.store.records()}),
            "vector_backend": self.store.backend,
            "embedding_provider": self.embedder.name,
            "retrieval_mode": self.settings.retrieval_mode,
            "chunk_strategy": self.settings.chunk_strategy,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }

    # ========================================================================
    # 检索
    # ========================================================================

    async def search(
        self, query: str, top_k: int | None = None, mode: str | None = None
    ) -> list[RetrievedChunk]:
        if not self.is_indexed:
            await self.index()
        return await self.retriever.retrieve(query, top_k=top_k, mode=mode)

    # ========================================================================
    # 问答
    # ========================================================================

    async def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        chunks = await self.search(question, top_k=top_k)

        if not chunks:
            return RagAnswer(
                answer="资料库中还没有内容，或没有检索到相关片段。请先建立索引。",
                sources=[],
                used_context=False,
            )

        prompt = self._build_prompt(question, chunks)
        reply = await self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,  # 事实型问答要低温度，减少发挥
        )
        return RagAnswer(answer=reply.content, sources=chunks, used_context=True)

    async def answer_stream(self, question: str, top_k: int | None = None):
        """流式版：先 yield 来源，再逐 token yield 答案。

        为什么要先给来源？因为用户看到「检索到了哪几段」比看到答案更能建立信任，
        而且检索本来就在生成之前完成，顺手就给了。
        """
        chunks = await self.search(question, top_k=top_k)
        yield ("sources", chunks)

        if not chunks:
            yield ("token", "资料库中还没有内容，或没有检索到相关片段。")
            return

        prompt = self._build_prompt(question, chunks)
        async for piece in self.llm.stream_chat([{"role": "user", "content": prompt}]):
            yield ("token", piece)

    # ========================================================================
    # Prompt 构造（防幻觉三道闸都在这里）
    # ========================================================================

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            loc = f"{c.source}" + (f" 第{c.page}页" if c.page else "")
            blocks.append(f"[{i}] （来源：{loc}）\n{c.text}")

        context = "\n\n".join(blocks)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"===== 资料开始 =====\n{context}\n===== 资料结束 =====\n\n"
            f"用户问题：{question}\n\n"
            "请依据上面的资料回答，并在引用处标注编号。"
        )


# ============================================================================
# 供 tools.search_docs 使用（避免 tools ↔ rag 循环 import）
# ============================================================================

_SERVICE: RagService | None = None


def register_service(service: RagService) -> None:
    """注册全局实例，并把它接到工具的检索钩子上。"""
    global _SERVICE
    _SERVICE = service

    def _hook(query: str, top_k: int) -> str:
        import asyncio

        if _SERVICE is None:
            raise RuntimeError("RAG 服务未初始化")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = _SERVICE.search(query, top_k=top_k)
        chunks = loop.run_until_complete(coro) if loop is None else asyncio.run_coroutine_threadsafe(coro, loop).result()

        if not chunks:
            return "未检索到相关文档片段。"

        return "\n\n".join(
            f"[{i}] （{c.source}{f' 第{c.page}页' if c.page else ''}）\n{c.text[:300]}"
            for i, c in enumerate(chunks, start=1)
        )

    tools.set_search_hook(_hook)
