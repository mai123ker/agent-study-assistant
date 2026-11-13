"""Embedding 提供方（Day 10）。

三种实现，靠配置切换：

| provider | 用途 | 代价 |
|---|---|---|
| `api` | 生产/演示用，走 OpenAI 兼容的 /embeddings | 要 key，按量计费 |
| `local` | 本地 sentence-transformers | 首次要下模型（几百 MB），但完全离线 |
| `hash` | **仅用于测试**，确定性伪向量 | 没有语义能力，只是让流程能跑通 |

把 embedding 抽象出来，是为了让「换模型」变成改一行配置 ——
面试问「你怎么选 embedding 模型」，你可以答：
中文场景优先 bge 系列；有 API 配额就用 API；离线要求高就本地部署。
**关键是要能说出你的评测方法**，而不是背模型名。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import Protocol

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    dim: int
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


# ============================================================================
# 离线确定性伪向量（测试用）
# ============================================================================

class HashEmbedding:
    """把文本哈希进固定维度向量。

    用字符三元组做特征，所以**字面相近的文本向量也相近** ——
    这足以让检索流程和测试跑通，但它不懂语义。

    ⚠️ 不要在真实演示里用它。面试官问「语义相近但用词不同的句子能召回吗」，
    用它就答不上来。生产请切 `api` 或 `local`。
    """

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        padded = f"^{text}$"
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            h = int(hashlib.md5(tri.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % self.dim] += 1.0
        return _l2_normalize(vec)


# ============================================================================
# API（OpenAI 兼容）
# ============================================================================

class ApiEmbedding:
    """走 OpenAI 兼容的 /embeddings 接口。

    阿里云百炼的 text-embedding-v3、智谱、硅基流动都兼容这个协议，
    所以只改 base_url 和 model 就能换。
    """

    name = "api"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dim = 1024  # 首次调用后会被真实维度覆盖

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.embedding_key:
            raise RuntimeError(
                "embedding_provider=api 但没配 key。"
                "请在 .env 里设置 LLM_API_KEY 或 EMBEDDING_API_KEY"
            )

        url = f"{self.settings.embedding_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.embedding_key}",
            "Content-Type": "application/json",
        }

        out: list[list[float]] = []
        # 接口通常限制单次条数，分批送
        batch = 16
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            for i in range(0, len(texts), batch):
                resp = await client.post(
                    url,
                    headers=headers,
                    json={"model": self.settings.embedding_model, "input": texts[i : i + batch]},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda d: d.get("index", 0))
                out.extend(item["embedding"] for item in data)

        if out:
            self.dim = len(out[0])
        return out


# ============================================================================
# 本地模型
# ============================================================================

class LocalEmbedding:
    """sentence-transformers 本地模型。

    需要额外安装：pip install sentence-transformers
    中文推荐 BAAI/bge-small-zh-v1.5（约 100MB，CPU 也能跑）。
    """

    name = "local"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 512

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "未安装 sentence-transformers。"
                    "执行：pip install sentence-transformers，"
                    "或把 EMBEDDING_PROVIDER 改回 api / hash"
                ) from exc
            logger.info("加载本地 embedding 模型: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def _run() -> list[list[float]]:
            model = self._load()
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [v.tolist() for v in vecs]

        # 模型推理是同步阻塞的，丢到线程里，别卡住事件循环
        return await asyncio.to_thread(_run)


# ============================================================================
# 工厂
# ============================================================================

def build_embedding(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "api":
        return ApiEmbedding(settings)
    if provider == "local":
        return LocalEmbedding()
    if provider == "hash":
        return HashEmbedding(settings.embedding_dim)

    raise ValueError(
        f"未知 embedding_provider `{provider}`，可选：api / local / hash"
    )


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。向量已归一化时等价于内积，但这里写成通用版。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
