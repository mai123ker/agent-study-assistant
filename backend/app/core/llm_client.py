"""LLM 调用封装（Day 2 多轮 / Day 4 工具 / Day 18 流式 / Day 19 工程健壮性）。

Day 1 时它只吃一个字符串。现在它承担四件事，但**始终只在这一个文件里**：
后续加缓存、加多 provider 路由、加成本统计，都只改这里，不动路由代码。

设计原则：把「不可靠的外部依赖」关在一个笼子里。
模型接口会超时、会 429、会 5xx、会返回畸形 JSON —— 这些都不该让服务崩。

面试要点：
- 哪些错该重试？超时 / 连接失败 / 429 / 5xx → 重试。
  哪些不该？400 / 401 / 403 / 404 → 重试一万次也一样，直接失败。
  这个区分是「你有没有真在生产里踩过」的分水岭。
- 为什么用指数退避？对方限流时你立刻重试只会加剧限流。
- 流式的价值是什么？用户感知的是**首字延迟**，不是总延迟。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

Message = dict[str, Any]


# ============================================================================
# 错误类型
# ============================================================================

class LLMError(Exception):
    """统一错误。retryable 决定要不要重试 —— 这是本文件最重要的一个字段。"""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，**不要在这里就 json.loads**，交给 tools 层校验

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    latency_ms: int = 0
    retries: int = 0

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

    def as_assistant_message(self) -> Message:
        """转回 messages 里 role=assistant 的那条（带 tool_calls）。"""
        msg: Message = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [tc.as_dict() for tc in self.tool_calls]
        return msg


# ============================================================================
# 客户端
# ============================================================================

class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ---------- 元信息 ----------

    @property
    def model_name(self) -> str:
        return "mock" if not self.settings.use_real_llm else self.settings.llm_model

    @property
    def is_mock(self) -> bool:
        return not self.settings.use_real_llm

    # ========================================================================
    # 主入口：一次完整调用（含重试）
    # ========================================================================

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMReply:
        start = time.perf_counter()

        if self.is_mock:
            reply = self._mock_chat(messages, tools)
            reply.latency_ms = int((time.perf_counter() - start) * 1000)
            return reply

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data, retries = await self._post_with_retry(payload)
        choice = data["choices"][0]["message"]

        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            )
            for i, tc in enumerate(choice.get("tool_calls") or [])
        ]

        return LLMReply(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            model=self.settings.llm_model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            retries=retries,
        )

    # ========================================================================
    # 结构化输出（Day 3）：JSON 解析失败要兜底，不能崩
    # ========================================================================

    async def chat_json(
        self,
        messages: list[Message],
        schema_hint: str = "",
    ) -> tuple[dict[str, Any] | None, str | None]:
        """返回 (解析结果, 错误信息)。失败时 result 为 None，**不抛异常**。"""
        if schema_hint:
            messages = [
                *messages,
                {"role": "system", "content": f"只输出 JSON，不要任何解释文字。结构：{schema_hint}"},
            ]

        reply = await self.chat(messages, json_mode=True)
        return self._parse_json(reply.content)

    @staticmethod
    def _parse_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
        """模型返回的 JSON 有四种常见坏法，逐个兜。

        ① 直接就是合法 JSON
        ② 被 ```json ... ``` 代码块包着
        ③ 前后加了「好的，这是结果：」之类的解释文字
        ④ 真的语法错误 → 放弃，返回错误
        """
        if not text or not text.strip():
            return None, "模型返回为空"

        stripped = text.strip()

        # 先看整体是不是合法 JSON。
        # 如果是、但不是对象（数组/字符串/null），**明确报错**而不是从数组里
        # 「捞」出第一个元素 —— 那会悄悄改变数据形状，是最难查的一类 bug。
        try:
            whole = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(whole, dict):
                return whole, None
            return None, f"顶层不是 JSON 对象（收到 {type(whole).__name__}）"

        candidates = [stripped]

        # ② 剥代码块
        fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if fence:
            candidates.append(fence.group(1))

        # ③ 截取第一个 { 到最后一个 }（处理「好的，结果是：{...}」这类前后夹话）
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            candidates.append(text[first : last + 1])

        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed, None

        return None, f"无法解析为 JSON（前 120 字：{text[:120]}）"

    # ========================================================================
    # 流式（Day 18）
    # ========================================================================

    async def stream_chat(self, messages: list[Message]) -> AsyncIterator[str]:
        """逐 token 产出文本。调用方包成 SSE 吐给前端。"""
        if self.is_mock:
            async for piece in self._mock_stream(messages):
                yield piece
            return

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            async with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise LLMError(
                        f"模型接口返回 {resp.status_code}: {body[:300]}",
                        retryable=resp.status_code >= 500 or resp.status_code == 429,
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece

    # ========================================================================
    # 内部：带重试的 POST
    # ========================================================================

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _timeout(self) -> httpx.Timeout:
        """连接超时和读超时要分开设。

        连接超时短（TCP 握手，不通就是不通）；读超时长（模型生成本来就慢）。
        合成一个 30s 的话，一个连不通的地址要干等 30 秒 —— 面试可以拿这个说。
        """
        return httpx.Timeout(
            connect=self.settings.connect_timeout,
            read=self.settings.request_timeout,
            write=self.settings.request_timeout,
            pool=self.settings.request_timeout,
        )

    async def _post_with_retry(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        max_attempts = self.settings.llm_max_retries + 1
        last_error: LLMError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout()) as client:
                    resp = await client.post(url, headers=self._headers(), json=payload)

                if resp.status_code >= 400:
                    raise self._classify_http_error(resp.status_code, resp.text)

                return resp.json(), attempt - 1

            except LLMError as exc:
                last_error = exc
                if not exc.retryable or attempt == max_attempts:
                    raise
                await self._sleep_before_retry(attempt, max_attempts, str(exc))

            except httpx.TimeoutException as exc:
                last_error = LLMError(f"请求超时: {exc}", retryable=True)
                if attempt == max_attempts:
                    raise last_error from exc
                await self._sleep_before_retry(attempt, max_attempts, "请求超时")

            except httpx.HTTPError as exc:
                last_error = LLMError(f"网络错误: {exc}", retryable=True)
                if attempt == max_attempts:
                    raise last_error from exc
                await self._sleep_before_retry(attempt, max_attempts, f"网络错误 {exc}")

        raise last_error or LLMError("未知错误", retryable=False)

    async def _sleep_before_retry(self, attempt: int, max_attempts: int, reason: str) -> None:
        delay = self._backoff_delay(attempt)
        logger.warning(
            "LLM 调用失败（第 %d/%d 次）：%s —— %.2fs 后重试",
            attempt, max_attempts, reason, delay,
        )
        await asyncio.sleep(delay)

    @staticmethod
    def _classify_http_error(status: int, body: str) -> LLMError:
        """决定这个错值不值得重试 —— 面试高频。"""
        # 429 限流、5xx 服务端问题：重试有意义
        retryable = status == 429 or status >= 500
        # 400 参数错、401 key 错、403 无权限、404 路径错：重试一万次也一样
        return LLMError(
            f"模型接口返回 {status}: {body[:300]}",
            retryable=retryable,
            status_code=status,
        )

    def _backoff_delay(self, attempt: int) -> float:
        """指数退避 + 上限。attempt 从 1 开始。"""
        delay = self.settings.llm_retry_base_delay * (2 ** (attempt - 1))
        return min(delay, self.settings.llm_retry_max_delay)

    # ========================================================================
    # mock：离线可跑，且能模拟工具调用（让 Day 4/5 的测试不依赖真实 API）
    # ========================================================================

    _TIME_HINTS = ("几点", "时间", "日期", "现在", "today", "time")
    _DOC_HINTS = ("文档", "资料", "笔记", "pdf", "PDF", "检索", "根据资料")

    def _mock_chat(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> LLMReply:
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        tool_names = {t["function"]["name"] for t in (tools or [])}
        already_called = any(m.get("role") == "tool" for m in messages)

        # RAG 场景：提示词里带了检索到的资料，mock 也要装得像一点，
        # 否则会把整段 prompt 原样回显，演示时很难看
        if "===== 资料开始 =====" in last_user:
            return LLMReply(content=self._mock_rag_answer(last_user), model="mock")

        # 已经拿到工具结果 → 生成最终回答（ReAct 循环的出口）
        if already_called:
            tool_output = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "tool"),
                "",
            )
            return LLMReply(content=f"[MOCK] 根据工具返回的结果：{tool_output}", model="mock")

        # 还没调工具 → 判断该不该调
        if tool_names:
            name, args = self._pick_mock_tool(last_user, tool_names)
            if name:
                return LLMReply(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"mock_call_{name}",
                            name=name,
                            arguments=json.dumps(args, ensure_ascii=False),
                        )
                    ],
                    model="mock",
                )

        turns = sum(1 for m in messages if m.get("role") in ("user", "assistant")) // 2
        return LLMReply(
            content=(
                f"[MOCK 模式] 收到你的问题：{last_user}\n\n"
                f"本次上下文包含 {len(messages)} 条消息（约 {turns} 轮历史）。\n"
                "整条链路已打通：Vue → FastAPI → LLMClient → 返回。\n"
                "接真实模型：把 backend/.env 的 LLM_PROVIDER 改成 openai 并填 LLM_API_KEY。"
            ),
            model="mock",
        )

    @staticmethod
    def _mock_rag_answer(prompt: str) -> str:
        """从 RAG 提示词里抽出资料块，拼一个像样的 mock 回答。

        目的只是让 mock 模式下的演示不难看，**它没有任何语义能力**。
        接入真实模型后这段逻辑不会被触发。
        """
        question = ""
        if "用户问题：" in prompt:
            question = prompt.split("用户问题：", 1)[1].split("\n", 1)[0].strip()

        body = prompt.split("===== 资料开始 =====", 1)[1].split("===== 资料结束 =====", 1)[0]
        blocks = [b.strip() for b in body.split("[") if b.strip()]

        if not blocks:
            return "资料中没有相关内容。"

        first = blocks[0]
        preview = first.split("）", 1)[-1].strip()[:220].replace("\n", " ")

        return (
            f"[MOCK 模式] 针对「{question}」，检索到 {len(blocks)} 段资料。\n\n"
            f"资料 [1] 的主要内容是：{preview}…… [1]\n\n"
            "（当前是 mock 模式，回答由检索结果拼接而成。"
            "接入真实模型后，这里会是对资料的自然语言总结，并标注引用编号。）"
        )

    @staticmethod
    def _pick_mock_tool(question: str, available: set[str]) -> tuple[str | None, dict[str, Any]]:
        if "calculator" in available:
            m = re.search(r"\d+(?:\.\d+)?\s*[\+\-\*/×]\s*\d+(?:\.\d+)?", question)
            if m:
                return "calculator", {"expression": m.group(0).replace("×", "*")}

        if "get_current_time" in available and any(h in question for h in LLMClient._TIME_HINTS):
            return "get_current_time", {"timezone": "Asia/Shanghai"}

        if "search_docs" in available and any(h in question for h in LLMClient._DOC_HINTS):
            return "search_docs", {"query": question, "top_k": 3}

        return None, {}

    async def _mock_stream(self, messages: list[Message]) -> AsyncIterator[str]:
        reply = self._mock_chat(messages, None)
        for ch in reply.content:
            await asyncio.sleep(0.004)
            yield ch
