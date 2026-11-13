"""请求 / 响应模型。

关键认知：用 Pydantic 把「接口契约」写死。
前端传什么、后端返回什么一眼可见，还自动生成 /docs 文档。

Day 1 只有 message/session_id；现在扩到三种模式（chat / rag / agent），
并支持流式。**改契约就是改这里** —— 前端对着 /docs 就能写。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ChatMode = Literal["chat", "rag", "agent"]


# ============================================================================
# 对话
# ============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="用户输入的问题")
    session_id: str | None = Field(
        default=None,
        description="会话 ID。不传则后端新建并回传，后续带上即可获得多轮记忆",
    )
    mode: ChatMode = Field(
        default="chat",
        description="chat=普通对话；rag=基于本地资料回答；agent=带工具调用的 ReAct 循环",
    )
    top_k: int | None = Field(default=None, ge=1, le=10, description="RAG 模式检索条数，默认取配置值")


class SourceRef(BaseModel):
    """RAG 回答引用的来源。有出处 = 可核查 = 面试加分。"""

    index: int = Field(..., description="引用序号，对应正文里的 [1] [2]")
    source: str = Field(..., description="来源文件")
    page: int | None = Field(default=None, description="页码（PDF 才有）")
    score: float = Field(..., description="融合后的相关度得分")
    preview: str = Field(..., description="片段预览")


class ToolTrace(BaseModel):
    """Agent 模式的一步工具调用记录 —— 可观测性的最小形态。"""

    step: int
    tool: str
    arguments: str
    ok: bool
    result: str
    duration_ms: int


class ChatResponse(BaseModel):
    reply: str = Field(..., description="模型回复")
    model: str = Field(..., description="本次使用的模型/模式标识")
    latency_ms: int = Field(..., ge=0, description="端到端耗时（毫秒）")
    session_id: str = Field(..., description="会话 ID，前端必须存下来并回传")
    mode: ChatMode = Field(default="chat", description="本次实际使用的模式")
    sources: list[SourceRef] = Field(default_factory=list, description="RAG 模式的引用来源")
    trace: list[ToolTrace] = Field(default_factory=list, description="Agent 模式的工具调用轨迹")
    steps: int = Field(default=0, description="Agent 模式实际执行步数")
    stopped_by: str | None = Field(
        default=None,
        description="Agent 循环的终止原因：answer=拿到答案 / max_steps=超步数 / repeat=重复动作 / error=异常",
    )
    degraded: bool = Field(default=False, description="是否发生了降级（如模型调用失败后返回兜底回复）")


# ============================================================================
# 会话
# ============================================================================

class SessionInfo(BaseModel):
    session_id: str
    turns: int = Field(..., description="已保留的对话轮数")
    messages: int = Field(..., description="已保留的消息条数")


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool


# ============================================================================
# RAG / 文档
# ============================================================================

class IndexRequest(BaseModel):
    rebuild: bool = Field(default=False, description="是否清空后重建索引")


class IndexResponse(BaseModel):
    indexed_files: int
    total_chunks: int
    strategy: str
    chunk_size: int
    chunk_overlap: int
    elapsed_ms: int
    files: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=20)
    mode: Literal["vector", "bm25", "hybrid"] | None = Field(
        default=None, description="不传则用配置里的默认检索模式"
    )


class SearchHit(BaseModel):
    index: int
    source: str
    page: int | None = None
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
    elapsed_ms: int


class RagStatsResponse(BaseModel):
    indexed: bool
    chunks: int
    files: int
    vector_backend: str
    embedding_provider: str
    retrieval_mode: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int


# ============================================================================
# 工具
# ============================================================================

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    dangerous: bool


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]
    recent_calls: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# 系统
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    llm_provider: str
    llm_model: str
    rag_indexed: bool
