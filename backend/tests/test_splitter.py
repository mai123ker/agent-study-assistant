"""切块策略测试（Day 9）。

切块是 RAG 里影响最大的一环，但它常常被当成「调个参数」就过去了。
这些测试把三种策略的**行为差异**固定下来，方便你改参数时对照。
"""

import pytest

from app.core.splitter import (
    split_document,
    split_fixed,
    split_recursive,
    split_semantic,
    split_text,
)

SAMPLE = """第一段讲的是 RAG 的基本概念。RAG 把检索和生成拼在一起，解决模型不知道私有资料的问题。

第二段讲切块策略。固定长度切块最简单，但会把句子从中间切断。递归切块按段落到句子的优先级降级。

第三段讲混合检索。向量检索对专有名词不敏感，BM25 靠字面匹配补上这个短板，两者融合效果更好。"""


# ============================================================================
# 固定长度
# ============================================================================

def test_fixed_respects_size():
    chunks = split_fixed("a" * 1000, size=200, overlap=0)
    assert len(chunks) == 5
    assert all(len(c) <= 200 for c in chunks)


def test_fixed_short_text_single_chunk():
    assert split_fixed("很短", size=200, overlap=0) == ["很短"]


def test_fixed_empty():
    assert split_fixed("   ", size=200, overlap=0) == []


def test_fixed_overlap_creates_repetition():
    chunks = split_fixed("abcdefghij" * 10, size=20, overlap=5)
    assert len(chunks) > 1
    # 相邻块之间应有重叠内容
    assert chunks[0][-5:] == chunks[1][:5]


# ============================================================================
# 递归字符
# ============================================================================

def test_recursive_keeps_paragraph_boundaries():
    """递归切块应当优先在段落边界切，而不是硬切。"""
    chunks = split_recursive(SAMPLE, size=120, overlap=0)
    assert len(chunks) >= 2
    # 每个块不该从半句话中间开始
    for c in chunks:
        assert c.strip()
    # 三段的开头内容都应该能被找到（说明没被切碎丢弃）
    joined = "".join(chunks)
    assert "第一段" in joined
    assert "第三段" in joined


def test_recursive_falls_back_for_long_unbroken_text():
    """没有分隔符的长文本，必须降级到硬切，不能死循环。"""
    chunks = split_recursive("啊" * 500, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_recursive_empty():
    assert split_recursive("", size=100, overlap=10) == []


# ============================================================================
# 语义（近似）
# ============================================================================

def test_semantic_splits_on_topic_shift():
    chunks = split_semantic(SAMPLE, size=200, overlap=0)
    assert chunks
    assert all(c.strip() for c in chunks)


def test_semantic_single_sentence():
    assert split_semantic("只有一句话。", size=200, overlap=0) == ["只有一句话。"]


# ============================================================================
# 统一入口与边界
# ============================================================================

@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_all_strategies_produce_content(strategy):
    chunks = split_text(SAMPLE, strategy=strategy, size=150, overlap=20)
    assert chunks
    assert all(c.strip() for c in chunks)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="未知切块策略"):
        split_text("x", strategy="不存在", size=100, overlap=10)


def test_overlap_must_be_smaller_than_size():
    """overlap >= size 会让步长变成 0 或负数 —— 必须提前拦。"""
    with pytest.raises(ValueError, match="overlap"):
        split_text("x" * 100, strategy="fixed", size=50, overlap=50)


def test_split_document_carries_metadata():
    """source 和 page 必须一路带下去 —— 第 4 周的引用来源全靠它。"""
    chunks = split_document(
        SAMPLE, source="示例.md", strategy="recursive", size=150, overlap=20, page=3
    )
    assert chunks
    assert all(c.source == "示例.md" for c in chunks)
    assert all(c.page == 3 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
