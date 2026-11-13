"""工具接口（Day 4 / Day 20）。

为什么要暴露一个手动调工具的接口？
1. 演示 Day 20 的权限拦截时，可以当场构造一个越界路径看它被拒 —— 比讲一百遍都有说服力
2. 调工具的参数校验、异常兜底都能单独测，不用起整个 Agent 循环
3. `recent_calls` 让调用历史可见，是 Day 21 可观测性的最小形态

⚠️ 生产环境**不要**暴露这种手动调工具的接口 —— 它等于把工具直接开给了调用方。
这里保留是因为它是教学项目，且所有工具本身都有白名单和参数校验。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.tools import ToolRegistry
from app.deps import get_tool_registry
from app.schemas import ToolInfo, ToolListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["tools"])


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="工具参数，键名见 GET /api/tools"
    )


class ToolCallResponse(BaseModel):
    tool: str
    ok: bool
    result: str = ""
    error: str | None = None


@router.get("", response_model=ToolListResponse, summary="列出已注册工具")
async def list_tools(registry: ToolRegistry = Depends(get_tool_registry)) -> ToolListResponse:
    tools = [
        ToolInfo(
            name=spec.name,
            description=spec.description,
            parameters=spec.arg_model.model_json_schema(),
            dangerous=spec.dangerous,
        )
        for spec in (registry.get(n) for n in registry.names())
        if spec is not None
    ]
    return ToolListResponse(tools=tools, recent_calls=registry.call_log[-20:])


@router.post("/{name}/call", response_model=ToolCallResponse, summary="手动调用一个工具")
async def call_tool(
    name: str,
    req: ToolCallRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
) -> ToolCallResponse:
    """注意：`registry.execute` 内部会兜住所有异常，所以这里永远不会 500。

    这正是设计目标 —— 工具出错是**正常业务流**，不是服务器故障。
    失败信息会原样回给调用方（在 Agent 里就是回给模型）。
    """
    result = registry.execute(name, req.arguments)
    return ToolCallResponse(
        tool=name,
        ok=result.ok,
        result=result.content,
        error=result.error,
    )
