"""手写 ReAct 循环（Day 5）—— 本周分水岭。

**不用 LangChain、不用 LangGraph。** 理由很简单：
面试官问「Agent 循环怎么终止」，你答"框架帮我处理的"，这一题就废了。
手写一遍你会发现，所谓 Agent 循环，本质就是一个带终止条件的 while。

ReAct = Reasoning + Acting：模型每一步先想（要不要调工具、调哪个），
再行动（我们执行工具），把结果喂回去，继续下一轮。

### 三个终止条件，缺一不可

| 终止条件 | 不做的后果 |
|---|---|
| ① 拿到最终答案（模型不再请求工具） | 正常出口，必须有 |
| ② 达到 `max_steps` | 模型陷入无效循环 → **无限烧钱**，请求永不返回 |
| ③ 重复动作检测 | 模型反复用同样参数调同一个工具 → ②能兜住但白烧 N 次 token |

第 ③ 条是加分项。做了它，你就能说「我考虑过死循环，而且不只靠步数上限兜底」。

### 为什么要存 trace？

因为 Agent 是**非确定性**的：同一个问题两次跑，走的路径可能不同。
出问题时没有 trace 就只能靠猜。这也是 Day 21 可观测性的雏形。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.core.llm_client import LLMClient, LLMError, Message
from app.core.tools import ToolRegistry

logger = logging.getLogger(__name__)


REACT_SYSTEM_PROMPT = """你是一个会使用工具的助手。你可以调用工具来获取你不知道的信息。

工作方式：
1. 先判断这个问题是否需要工具。**如果不需要，直接回答，不要调用任何工具。**
2. 需要工具时，可以一次调用一个或多个。
3. 拿到工具结果后，判断信息是否足够回答。足够就直接给出答案；不够可以继续调用。
4. 工具报错时，不要重复同样的调用 —— 换个参数，或者直接告诉用户你做不到。

重要：闲聊、常识、解释概念这类问题不需要工具，直接回答。"""


@dataclass
class AgentStep:
    step: int
    tool: str
    arguments: str
    ok: bool
    result: str
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.result[:500],
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentRun:
    answer: str
    steps: int = 0
    trace: list[AgentStep] = field(default_factory=list)
    stopped_by: str = "answer"  # answer | max_steps | repeat | error
    degraded: bool = False

    @property
    def used_tools(self) -> bool:
        return bool(self.trace)


class ReactAgent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or get_settings()
        self.max_steps = self.settings.agent_max_steps

    async def run(
        self,
        question: str,
        history: list[Message] | None = None,
        max_steps: int | None = None,
    ) -> AgentRun:
        limit = max_steps or self.max_steps

        messages: list[Message] = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        tool_schemas = self.registry.schemas()
        trace: list[AgentStep] = []
        last_signature: tuple | None = None

        for step in range(1, limit + 1):
            # ---------- ① 思考：模型决定下一步 ----------
            try:
                reply = await self.llm.chat(messages, tools=tool_schemas)
            except LLMError as exc:
                logger.error("Agent 第 %d 步模型调用失败: %s", step, exc)
                return AgentRun(
                    answer=f"模型调用失败：{exc}。请稍后重试或检查 LLM 配置。",
                    steps=step - 1,
                    trace=trace,
                    stopped_by="error",
                    degraded=True,
                )

            # ---------- 终止条件 ①：不请求工具 = 给出最终答案 ----------
            if not reply.wants_tool:
                return AgentRun(
                    answer=reply.content or "（模型没有返回内容）",
                    steps=step - 1,
                    trace=trace,
                    stopped_by="answer",
                )

            # ---------- 终止条件 ③：重复动作检测 ----------
            signature = tuple(
                sorted((tc.name, self._normalize_args(tc.arguments)) for tc in reply.tool_calls)
            )
            if signature == last_signature:
                logger.warning("检测到重复工具调用，提前终止（第 %d 步）", step)
                return AgentRun(
                    answer=(
                        "检测到我在重复同样的工具调用，已主动停止以避免死循环。"
                        "请把问题描述得更具体一些，或换个问法。"
                    ),
                    steps=step - 1,
                    trace=trace,
                    stopped_by="repeat",
                    degraded=True,
                )
            last_signature = signature

            # ---------- ② 行动：把 assistant 的 tool_calls 记进上下文 ----------
            messages.append(reply.as_assistant_message())

            for tc in reply.tool_calls:
                started = time.perf_counter()
                result = self.registry.execute(tc.name, tc.arguments)
                elapsed = int((time.perf_counter() - started) * 1000)

                # 工具失败也照样回填给模型，让它自己决定下一步 —— 这是关键设计
                messages.append(result.as_tool_message(tc.id))
                trace.append(
                    AgentStep(
                        step=step,
                        tool=tc.name,
                        arguments=tc.arguments,
                        ok=result.ok,
                        result=result.content if result.ok else (result.error or ""),
                        duration_ms=elapsed,
                    )
                )
                logger.info(
                    "Agent step %d | tool=%s | ok=%s | %dms",
                    step, tc.name, result.ok, elapsed,
                )

        # ---------- 终止条件 ②：达到最大步数 ----------
        logger.warning("Agent 达到最大步数 %d 仍未给出答案", limit)
        return AgentRun(
            answer=(
                f"我已经连续执行了 {limit} 步仍未得到确定答案，为避免无限制调用已停止。"
                "你可以把问题拆得更小一些再问我。"
            ),
            steps=limit,
            trace=trace,
            stopped_by="max_steps",
            degraded=True,
        )

    @staticmethod
    def _normalize_args(arguments: str) -> str:
        """把参数 JSON 规范化，避免 key 顺序不同导致误判为「不重复」。"""
        try:
            return json.dumps(json.loads(arguments), sort_keys=True, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return arguments
