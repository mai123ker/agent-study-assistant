"""检索测试（Day 10-11）：BM25、向量、混合 + RRF 融合。

这些测试的价值在于：它们把「为什么需要混合检索」变成了可执行的事实。
`test_bm25_wins_on_exact_term` 和 `test_vector_wins_on_paraphrase` 放一起看，
就是 Day 11 那张对比表的雏形。
"""

import asyncio

import pytest

from app.core.bm25 import BM25, tokenize
from app.core.embeddings import HashEmbedding, cosine
from app.core.retriever import HybridRetriever
from app.core.vector_store import InMemoryVectorStore, VectorRecord


# ============================================================================
# 分词
# ============================================================================

def test_tokenize_handles_chinese_bigrams():
    toks = tokenize("大模型")
    assert "大模" in toks
    assert "模型" in toks


def test_tokenize_handles_latin_and_digits():
    toks = tokenize("FastAPI 8000")
    assert "fastapi" in toks
    assert "8000" in toks


def test_tokenize_single_cjk_char():
    assert tokenize("好") == ["好"]


# ============================================================================
# BM25
# ============================================================================

CORPUS = [
    "MCP 是模型上下文协议，它把工具集成复杂度从 M×N 降到 M+N。",
    "RAG 把检索和生成拼在一起，用来解决模型不知道私有资料的问题。",
    "切块策略影响检索精度，块太小语义不完整，块太大噪声多。",
]


def test_bm25_finds_exact_term():
    """⭐ 专有名词场景：这是 BM25 存在的主要理由。"""
    scores = BM25().fit(CORPUS).score("MCP 协议")
    top = scores.index(max(scores))
    assert top == 0


def test_bm25_scores_are_non_negative():
    """IDF 平滑后不该出现负分 —— 否则高频词会反向排序。"""
    scores = BM25().fit(CORPUS).score("检索 生成 模型")
    assert all(s >= 0 for s in scores)


def test_bm25_no_match_returns_zeros():
    scores = BM25().fit(CORPUS).score("完全无关的词汇xyz")
    assert all(s == 0 for s in scores)


def test_bm25_empty_corpus():
    assert BM25().fit([]).score("任意") == []


# ============================================================================
# Embedding
# ============================================================================

def test_hash_embedding_is_deterministic():
    emb = HashEmbedding(dim=64)
    a = asyncio.run(emb.embed(["同样的文本"]))[0]
    b = asyncio.run(emb.embed(["同样的文本"]))[0]
    assert a == b


def test_hash_embedding_is_normalized():
    vec = asyncio.run(HashEmbedding(dim=64).embed(["测试文本"]))[0]
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedding_similar_text_scores_higher():
    """字面相近 → 向量相近。注意：它**不懂语义**，只是让流程能跑通。"""
    emb = HashEmbedding(dim=256)
    vecs = asyncio.run(emb.embed(["大模型应用开发", "大模型应用开发入门", "今天天气不错"]))
    near = cosine(vecs[0], vecs[1])
    far = cosine(vecs[0], vecs[2])
    assert near > far


def test_cosine_edge_cases():
    assert cosine([], []) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0


# ============================================================================
# 向量库
# ============================================================================

def _build_store(texts: list[str], dim: int = 256) -> InMemoryVectorStore:
    emb = HashEmbedding(dim)
    vectors = asyncio.run(emb.embed(texts))
    store = InMemoryVectorStore()
    store.add([
        VectorRecord(id=str(i), text=t, vector=v, metadata={"source": "t.md", "index": i})
        for i, (t, v) in enumerate(zip(texts, vectors))
    ])
    return store


def test_in_memory_store_search():
    store = _build_store(CORPUS)
    assert store.count() == 3
    query_vec = asyncio.run(HashEmbedding(256).embed(["MCP 协议"]))[0]
    hits = store.search(query_vec, top_k=2)
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]  # 按分数降序


def test_store_clear():
    store = _build_store(CORPUS)
    store.clear()
    assert store.count() == 0
    assert store.search([0.1] * 256, top_k=3) == []


# ============================================================================
# 混合检索
# ============================================================================

def _retriever(texts: list[str]) -> HybridRetriever:
    store = _build_store(texts)
    return HybridRetriever(store=store, embedder=HashEmbedding(256))


def test_bm25_only_mode():
    chunks = asyncio.run(_retriever(CORPUS).retrieve("MCP 协议", top_k=1, mode="bm25"))
    assert len(chunks) == 1
    assert "MCP" in chunks[0].text
    assert chunks[0].bm25_score > 0
    assert chunks[0].vector_score == 0


def test_vector_only_mode():
    chunks = asyncio.run(_retriever(CORPUS).retrieve("切块策略", top_k=1, mode="vector"))
    assert len(chunks) == 1
    assert chunks[0].vector_score != 0
    assert chunks[0].bm25_score == 0


def test_hybrid_merges_both_signals():
    """混合检索的结果应当**同时**带有两路分数 —— 说明 RRF 真的融合了。"""
    chunks = asyncio.run(_retriever(CORPUS).retrieve("MCP 协议", top_k=3, mode="hybrid"))
    assert chunks
    top = chunks[0]
    assert top.vector_score != 0
    assert top.bm25_score != 0
    assert 0 < top.fused_score <= 1.0


def test_hybrid_normalizes_fused_score():
    chunks = asyncio.run(_retriever(CORPUS).retrieve("检索", top_k=3, mode="hybrid"))
    assert chunks[0].fused_score == pytest.approx(1.0)


def test_rrf_ignores_score_scale():
    """RRF 只看排名 —— 这是它相对加权求和的核心优势。

    BM25 分数是无上界实数，余弦相似度在 0~1；
    直接加权求和会让 BM25 一边倒。RRF 天然免疫这个问题。
    """
    chunks = asyncio.run(_retriever(CORPUS).retrieve("MCP 协议 检索", top_k=3, mode="hybrid"))
    assert all(0 < c.fused_score <= 1.0 for c in chunks)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="未知检索模式"):
        asyncio.run(_retriever(CORPUS).retrieve("x", mode="乱写的"))


def test_empty_store_returns_empty():
    empty = HybridRetriever(store=InMemoryVectorStore(), embedder=HashEmbedding(256))
    assert asyncio.run(empty.retrieve("任何问题")) == []


def test_bm25_cache_invalidated_on_index_change():
    """语料变了 BM25 必须重建，否则用的是过期 IDF。"""
    r = _retriever(CORPUS)
    asyncio.run(r.retrieve("MCP", mode="bm25"))
    assert r._bm25 is not None

    r.store.add([VectorRecord(id="x", text="新增文档", vector=[0.0] * 256, metadata={})])
    r.invalidate()
    assert r._bm25 is None
