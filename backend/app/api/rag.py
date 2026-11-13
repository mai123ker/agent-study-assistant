"""RAG 接口（Day 8-11）。

把索引、检索、状态三个动作暴露出来，好处是：
1. 演示时可以先展示「检索到了哪几段」，再展示生成 —— 比直接看答案有说服力
2. Day 24 跑评测时，可以直接调 `/api/rag/search` 算 Recall@K，不用起整个问答链路
3. 面试官问「检索效果怎么验证」，你能当场打开 /docs 演示

`/api/rag/search` 支持手动指定 `mode`，这是为了让你能做**对比实验**：
同一个 query 分别用 vector / bm25 / hybrid 跑一遍，把结果记下来 ——
这就是 Day 11 那张 Recall@K 对比表的原始数据。
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from app.core.rag import RagService
from app.deps import get_rag_service
from app.schemas import (
    IndexRequest,
    IndexResponse,
    RagStatsResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/index", response_model=IndexResponse, summary="建立/重建索引")
async def build_index(
    req: IndexRequest,
    rag: RagService = Depends(get_rag_service),
) -> IndexResponse:
    try:
        stats = await rag.index(rebuild=req.rebuild)
    except RuntimeError as exc:
        # 典型场景：embedding_provider=api 但没配 key
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndexResponse(
        indexed_files=stats.indexed_files,
        total_chunks=stats.total_chunks,
        strategy=stats.strategy,
        chunk_size=stats.chunk_size,
        chunk_overlap=stats.chunk_overlap,
        elapsed_ms=stats.elapsed_ms,
        files=stats.files,
    )


@router.post("/search", response_model=SearchResponse, summary="检索（可指定检索模式）")
async def search(
    req: SearchRequest,
    rag: RagService = Depends(get_rag_service),
) -> SearchResponse:
    start = time.perf_counter()
    try:
        chunks = await rag.search(req.query, top_k=req.top_k, mode=req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(
        query=req.query,
        mode=req.mode or rag.settings.retrieval_mode,
        hits=[
            SearchHit(
                index=i,
                source=c.source,
                page=c.page,
                text=c.text,
                vector_score=round(c.vector_score, 4),
                bm25_score=round(c.bm25_score, 4),
                fused_score=round(c.fused_score, 4),
            )
            for i, c in enumerate(chunks, start=1)
        ],
        elapsed_ms=int((time.perf_counter() - start) * 1000),
    )


@router.get("/stats", response_model=RagStatsResponse, summary="索引状态")
async def stats(rag: RagService = Depends(get_rag_service)) -> RagStatsResponse:
    return RagStatsResponse(**rag.stats())
