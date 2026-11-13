"""对话接口（Day 2 多轮 / Day 4-5 Agent / Day 18 流式）。

Day 1 只有一条单轮路径。现在按 `mode` 分三条：

| mode | 链路 | 什么时候用 |
|---|---|---|
| `chat` | 记忆 → 模型 | 普通多轮对话 |
| `rag`  | 记忆 → 检索 → 带引用生成 | 问「我的资料里怎么写」 |
| `agent`| ReAct 循环（可调工具） | 需要算数、查时间、检索资料 |

三条路径共用同一个 `ConversationMemory`，所以换模式不会丢上下文。

**流式接口为什么单独开一个端点而不是加参数？**
因为返回类型根本不同：一个是 `ChatResponse` JSON，一个是 `text/event-stream`。
塞进同一个端点会让 OpenAPI 文档没法看，前端也要写两套解析分支。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.llm_client import LLMClient, LLMError
from app.core.memory import ConversationMemory
from app.core.rag import RagService
from app.core.react_agent import ReactAgent
from app.deps import get_llm_client, get_memory, get_rag_service, get_react_agent
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
    SessionInfo,
    SourceRef,
    ToolTrace,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


# ============================================================================
# 主接口
# ============================================================================

@router.post("/chat", response_model=ChatResponse, summary="对话（chat / rag / agent）")
async def chat(
    req: ChatRequest,
    memory: ConversationMemory = Depends(get_memory),
    llm: LLMClient = Depends(get_llm_client),
    rag: RagService = Depends(get_rag_service),
    agent: ReactAgent = Depends(get_react_agent),
) -> ChatResponse:
    session_id = memory.ensure(req.session_id)
    history = memory.history(session_id)
    start = time.perf_counter()

    def elapsed() -> int:
        return int((time.perf_counter() - start) * 1000)

    logger.info("收到请求 mode=%s session=%s: %s", req.mode, session_id, req.message[:80])

    # ---- agent 模式：ReAct 循环 ----
    if req.mode == "agent":
        run = await agent.run(req.message, history=history)
        memory.append(session_id, "user", req.message)
        memory.append(session_id, "assistant", run.answer)

        return ChatResponse(
            reply=run.answer,
            model=f"{llm.model_name}+agent",
            latency_ms=elapsed(),
            session_id=session_id,
            mode="agent",
            trace=[
                ToolTrace(
                    step=s.step,
                    tool=s.tool,
                    arguments=s.arguments,
                    ok=s.ok,
                    result=s.result[:500],
                    duration_ms=s.duration_ms,
                )
                for s in run.trace
            ],
            steps=run.steps,
            stopped_by=run.stopped_by,
            degraded=run.degraded,
        )

    # ---- rag 模式：检索 + 带引用生成 ----
    if req.mode == "rag":
        try:
            result = await rag.answer(req.message, top_k=req.top_k)
        except LLMError as exc:
            return _degraded_response(session_id, exc)

        memory.append(session_id, "user", req.message)
        memory.append(session_id, "assistant", result.answer)

        return ChatResponse(
            reply=result.answer,
            model=f"{llm.model_name}+rag",
            latency_ms=elapsed(),
            session_id=session_id,
            mode="rag",
            sources=[
                SourceRef(
                    index=i,
                    source=c.source,
                    page=c.page,
                    score=round(c.fused_score, 4),
                    preview=c.text[:160],
                )
                for i, c in enumerate(result.sources, start=1)
            ],
        )

    # ---- chat 模式：普通多轮 ----
    messages = memory.build_context(session_id, req.message)
    try:
        reply = await llm.chat(messages)
    except LLMError as exc:
        logger.error("模型调用失败: %s", exc)
        return _degraded_response(session_id, exc)

    memory.append(session_id, "user", req.message)
    memory.append(session_id, "assistant", reply.content)

    return ChatResponse(
        reply=reply.content,
        model=reply.model,
        latency_ms=reply.latency_ms,
        session_id=session_id,
        mode="chat",
    )


# ============================================================================
# 流式（Day 18）
# ============================================================================

@router.post("/chat/stream", summary="流式对话（SSE）")
async def chat_stream(
    req: ChatRequest,
    memory: ConversationMemory = Depends(get_memory),
    llm: LLMClient = Depends(get_llm_client),
    rag: RagService = Depends(get_rag_service),
) -> StreamingResponse:
    session_id = memory.ensure(req.session_id)

    async def event_stream() -> AsyncIterator[str]:
        collected: list[str] = []

        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 先把 session_id 推给前端，后面每轮都要带回来
        yield sse({"type": "meta", "session_id": session_id, "model": llm.model_name})

        try:
            if req.mode == "rag":
                async for kind, payload in rag.answer_stream(req.message, top_k=req.top_k):
                    if kind == "sources":
                        yield sse({
                            "type": "sources",
                            "sources": [
                                {
                                    "index": i,
                                    "source": c.source,
                                    "page": c.page,
                                    "score": round(c.fused_score, 4),
                                }
                                for i, c in enumerate(payload, start=1)
                            ],
                        })
                    else:
                        collected.append(payload)
                        yield sse({"type": "token", "text": payload})
            else:
                messages = memory.build_context(session_id, req.message)
                async for piece in llm.stream_chat(messages):
                    collected.append(piece)
                    yield sse({"type": "token", "text": piece})

        except LLMError as exc:
            logger.error("流式调用失败: %s", exc)
            yield sse({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 — 流里绝不能裸抛，否则前端会卡在 loading
            logger.exception("流式过程异常")
            yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        else:
            answer = "".join(collected)
            memory.append(session_id, "user", req.message)
            memory.append(session_id, "assistant", answer)

        # 无论成功失败都发 done —— 前端靠它关掉 loading 状态
        yield sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防止 Nginx 缓冲导致「假流式」
        },
    )


# ============================================================================
# 会话管理
# ============================================================================

@router.get("/sessions/{session_id}", response_model=SessionInfo, summary="查看会话状态")
async def session_info(
    session_id: str,
    memory: ConversationMemory = Depends(get_memory),
) -> SessionInfo:
    history = memory.history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"会话不存在或已清空: {session_id}")
    return SessionInfo(
        session_id=session_id,
        turns=memory.turn_count(session_id),
        messages=len(history),
    )


@router.delete("/sessions/{session_id}", response_model=ClearSessionResponse, summary="清空会话")
async def clear_session(
    session_id: str,
    memory: ConversationMemory = Depends(get_memory),
) -> ClearSessionResponse:
    return ClearSessionResponse(session_id=session_id, cleared=memory.clear(session_id))


# ============================================================================
# 内部
# ============================================================================

def _degraded_response(session_id: str, exc: Exception) -> ChatResponse:
    """模型挂了也要给用户一个能看懂的回复，而不是 500。"""
    return ChatResponse(
        reply=(
            "抱歉，模型服务暂时不可用。\n"
            f"原因：{exc}\n"
            "请检查 .env 里的 LLM_BASE_URL / LLM_API_KEY，或稍后重试。"
        ),
        model="degraded",
        latency_ms=0,
        session_id=session_id,
        mode="chat",
        degraded=True,
    )
