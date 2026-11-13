"""集中配置。

所有可变参数都从这里读取，禁止在业务代码里散落硬编码。
`LLM_PROVIDER=mock` 时整条链路离线可跑（含 RAG 的离线 embedding）。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根：backend/
BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 应用 ----
    app_name: str = "Agent Study Assistant"
    app_version: str = "0.2.0"

    # ---- LLM ----
    # mock   : 本地模拟回复，不联网（默认，离线可跑）
    # openai : 调用 OpenAI 兼容接口（DeepSeek / 通义 / Ollama 都兼容）
    llm_provider: str = "mock"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3

    # ---- 超时（Day 19）----
    # 连接超时和读超时分开：TCP 握手不通就是不通，模型生成本来就慢
    connect_timeout: float = 5.0
    request_timeout: float = 60.0

    # ---- 重试（Day 19）----
    llm_max_retries: int = 2
    llm_retry_base_delay: float = 0.5
    llm_retry_max_delay: float = 8.0

    # ---- 会话记忆（Day 2 / Day 13）----
    memory_max_turns: int = 10

    # ---- Agent（Day 5）----
    agent_max_steps: int = 8

    # ---- RAG（Day 8-11）----
    rag_enabled: bool = True
    rag_top_k: int = 3
    rag_fetch_k: int = 10  # 粗排取多少条给精排/融合
    chunk_size: int = 400
    chunk_overlap: int = 50
    chunk_strategy: str = "recursive"  # fixed | recursive | semantic

    # ---- Embedding ----
    # api   : 调 OpenAI 兼容的 /embeddings（通义 text-embedding-v3 等）
    # local : sentence-transformers 本地模型（需额外安装，首次要下模型）
    # hash  : 离线确定性伪向量，**只用于测试**，没有语义能力
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-v3"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_dim: int = 256  # 仅 hash provider 使用

    # ---- 向量库（Day 10）----
    # memory : 纯 Python 内存实现，零依赖（开发/演示够用）
    # chroma : ChromaDB 持久化（数据量大时切）
    vector_backend: str = "memory"
    vector_persist_dir: str = str(BASE_DIR / "data" / "chroma")
    vector_collection: str = "study_docs"

    # ---- 检索（Day 11）----
    # vector | bm25 | hybrid
    retrieval_mode: str = "hybrid"
    hybrid_alpha: float = 0.5  # 0=纯 BM25，1=纯向量

    # ---- 工具安全（Day 20）----
    # read_file 只能读这个目录内的文件
    tool_allow_dir: str = str(BASE_DIR / "data" / "docs")

    # ---- 文档目录 ----
    docs_dir: str = str(BASE_DIR / "data" / "docs")

    # ---- CORS ----
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---------------- 派生属性 ----------------

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_real_llm(self) -> bool:
        return self.llm_provider != "mock" and bool(self.llm_api_key)

    @property
    def embedding_key(self) -> str:
        """embedding 没单独配 key 就复用 LLM 的。"""
        return self.embedding_api_key or self.llm_api_key

    @property
    def docs_path(self) -> Path:
        return Path(self.docs_dir)

    @property
    def allow_path(self) -> Path:
        return Path(self.tool_allow_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
