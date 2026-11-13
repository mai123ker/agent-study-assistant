"""文档切块（Day 9）。

切块是 RAG 里**影响最大、最容易被忽略**的一环。
面试被问「检索不准怎么办」，第一个该答的就是切块粒度，而不是换模型。

三种策略对应三种思路：

| 策略 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| fixed | 每 N 字硬切 | 实现最简单、块长均匀 | 会把句子从中间切断 |
| recursive | 段落 → 句子 → 字符，逐级降级 | 尽量保住语义边界 | 块长不均匀 |
| semantic | 按句子间语义相似度切 | 语义最完整 | 慢，且需要 embedding |

`overlap` 的意义：防止一句话正好落在切点上、两边都检索不到。
代价是索引体积和 token 消耗变大 —— 这是典型的 trade-off，面试要主动说。
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

# 逐级降级的分隔符：优先在段落边界切，实在不行才切到字符
_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])")


@dataclass
class Chunk:
    """一个待入库的片段。

    `source` 和 `page` 必须一路带到最后 —— Day 25 做引用来源全靠它们。
    """

    text: str
    source: str
    page: int | None = None
    index: int = 0
    meta: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {"source": self.source, "page": self.page, "index": self.index}


# ============================================================================
# 策略一：固定长度
# ============================================================================

def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """每 size 字切一刀。最简单，但会切断句子。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


# ============================================================================
# 策略二：递归字符（默认）
# ============================================================================

def split_recursive(
    text: str,
    size: int,
    overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """按 段落 → 句子 → 字符 的优先级逐级降级切分。

    这是 LangChain `RecursiveCharacterTextSplitter` 的核心思路 ——
    名字很唬人，其实就是「优先在大的语义边界切，切不动了再降一级」。
    """
    seps = separators if separators is not None else _DEFAULT_SEPARATORS
    pieces = _recursive_split(text, size, seps, depth=0)
    return _merge_with_overlap(pieces, size, overlap)


def _recursive_split(text: str, size: int, seps: list[str], depth: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    if depth >= len(seps):
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep = seps[depth]
    if not sep or sep not in text:
        return _recursive_split(text, size, seps, depth + 1)

    parts = [p for p in text.split(sep) if p.strip()]
    out: list[str] = []
    for p in parts:
        out.extend(_recursive_split(p, size, seps, depth + 1))
    return out


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    """把碎片合并到接近 size，并在相邻块之间保留 overlap 个字符。

    关键约束：**输出块的长度必须始终 ≤ size**。

    容易踩的坑：新块开头拼接上一块的 overlap 尾巴后，如果不管剩余预算，
    块就会超出 size（100 字的块变 111 字）。块长超预期会打乱 token 预算，
    而且越滚越大。所以这里用工作队列，把放不下的部分重新排队切。
    """
    chunks: list[str] = []
    queue: deque[str] = deque(p for p in pieces if p)
    buf = ""

    while queue:
        piece = queue.popleft()

        # 单块本身就超限 → 先把它切成 size 的整数块，余下部分重新排队
        if len(piece) > size:
            queue.appendleft(piece[size:])
            piece = piece[:size]

        if not buf:
            buf = piece
            continue

        # 放得下就继续拼
        if len(buf) + len(piece) + 1 <= size:
            buf = f"{buf}\n{piece}"
            continue

        # 放不下 → 收口当前块，用 overlap 尾巴开新块
        chunks.append(buf)
        tail = buf[-overlap:] if overlap > 0 else ""
        separator = "\n" if tail else ""
        budget = size - len(tail) - len(separator)

        if budget <= 0:
            # overlap 已经吃掉全部预算，只能丢弃尾巴
            buf = piece
            continue

        if len(piece) > budget:
            queue.appendleft(piece[budget:])
            piece = piece[:budget]

        buf = f"{tail}{separator}{piece}"

    if buf.strip():
        chunks.append(buf)
    return chunks


# ============================================================================
# 策略三：语义切块（近似实现）
# ============================================================================

def split_semantic(text: str, size: int, overlap: int, threshold: float = 0.12) -> list[str]:
    """按相邻句子的「话题漂移」切分。

    真实语义切块要算句向量余弦相似度，需要 embedding 模型。
    这里用字符级 Jaccard 相似度做**近似**：相邻句用词重合度骤降，
    说明话题换了，就该在此处切一刀。

    局限要说清楚：这是廉价代理，不是真语义。
    面试被追问时主动承认，比被拆穿好。
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    groups: list[list[str]] = [[sentences[0]]]
    for prev, cur in zip(sentences, sentences[1:]):
        if _jaccard(prev, cur) < threshold and len("".join(groups[-1])) >= size * 0.5:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    # 再按 size 做一次兜底合并/切分，避免某个组过大
    out: list[str] = []
    for g in groups:
        joined = "".join(g)
        if len(joined) <= size:
            out.append(joined)
        else:
            out.extend(split_fixed(joined, size, overlap))
    return _merge_with_overlap(out, size, overlap)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ============================================================================
# 统一入口
# ============================================================================

_STRATEGIES = {
    "fixed": split_fixed,
    "recursive": split_recursive,
    "semantic": split_semantic,
}


def split_text(text: str, strategy: str, size: int, overlap: int) -> list[str]:
    fn = _STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"未知切块策略 `{strategy}`，可选：{', '.join(_STRATEGIES)}")
    if overlap >= size:
        raise ValueError(f"overlap({overlap}) 必须小于 size({size})，否则会无限循环")
    return fn(text, size, overlap)


def split_document(
    text: str,
    source: str,
    strategy: str,
    size: int,
    overlap: int,
    page: int | None = None,
) -> list[Chunk]:
    """把一个文档切成 Chunk 列表。"""
    pieces = split_text(text, strategy, size, overlap)
    return [
        Chunk(text=p, source=source, page=page, index=i)
        for i, p in enumerate(pieces)
        if p.strip()
    ]
