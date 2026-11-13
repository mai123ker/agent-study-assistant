"""BM25 关键词检索（Day 11）。

为什么不能只用向量检索？
因为向量检索对**专有名词**很弱 —— 你问「MCP 协议是什么」，
它可能给你一堆讲「协议」的泛泛段落，却漏掉真正写着 "MCP" 的那一段。
BM25 靠字面匹配，正好补这个短板。反过来，BM25 对「换个说法问」无能为力。
所以生产上基本都是**混合检索**。

为什么自己实现而不用 `rank_bm25`？
因为这个公式只有 20 行，手写一遍你就能回答面试官「BM25 的 k1 和 b 是什么」——
k1 控制词频饱和速度，b 控制文档长度归一化强度。用库你就只能背答案。

中文分词说明：这里没有引入 jieba，而是用 **CJK 字符二元组（bigram）**。
「大模型」→「大模」「模型」。这是工业界常用的免词典方案，
对检索任务效果接近分词，且零依赖。面试可以主动说这个取舍。
"""

from __future__ import annotations

import math
import re
from collections import Counter

_LATIN = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按单词，中文按字符二元组。"""
    tokens: list[str] = [m.group(0).lower() for m in _LATIN.finditer(text)]

    cjk_chars = _CJK.findall(text)
    if len(cjk_chars) == 1:
        tokens.append(cjk_chars[0])
    else:
        tokens.extend(cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1))

    return tokens


class BM25:
    """标准 BM25 Okapi。

    公式（每个 query 词累加）：
        IDF(q) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    - `k1`（默认 1.5）：词频饱和点。某个词出现 10 次不该比出现 1 次重要 10 倍。
    - `b`（默认 0.75）：长度归一化强度。长文档天然更容易命中词，要惩罚一下。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def fit(self, corpus: list[str]) -> "BM25":
        self._doc_tokens = []
        self._doc_len = []
        self._df = Counter()

        for doc in corpus:
            toks = tokenize(doc)
            counts = Counter(toks)
            self._doc_tokens.append(counts)
            self._doc_len.append(len(toks))
            for term in counts:
                self._df[term] += 1

        n = len(corpus)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        return self

    @property
    def size(self) -> int:
        return len(self._doc_tokens)

    def _idf(self, term: str) -> float:
        n = self.size
        if n == 0:
            return 0.0
        df = self._df.get(term, 0)
        # 加 1 的平滑形式，保证 IDF 非负（否则高频词会得到负分，反直觉）
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        """返回每个文档对 query 的得分（与语料顺序一致）。"""
        if not self._doc_tokens:
            return []

        q_terms = set(tokenize(query))
        scores = [0.0] * self.size

        for term in q_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, counts in enumerate(self._doc_tokens):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_len[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom

        return scores
