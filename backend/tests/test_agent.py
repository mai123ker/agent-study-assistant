"""ReAct 循环测试（Day 5）—— 三个终止条件逐个验证。

Day 5 是整个第一周的分水岭：**能不能不看框架手写 Agent 循环**。
这些测试就是那个「能」的证据。

用 `ScriptedLLM` 精确控制模型每一步返回什么，
所以每个终止分支都能被确定性地触发 —— 不用碰真实 API。
"""

import asyncio
import json
from typing import Any

import pytest

from app.config import Settings
from app.core.llm_client import LLMError, LLMReply, ToolCall
from app.core.react_agent import ReactAgent
from app.core.tools import build_default_registry


# ============================================================================
# 可编排的假模型
# ============================================================================

class ScriptedLLM:
    """按脚本返回回复，用来精确触发每个终止分支。"""

    model_name = "scripted"

    def __init__(self, replies: list[LLMReply | Exception]) -> None:
        self.replies = replies
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    async def chat(self, messages: list[dict], **kwargs: Any) -> LLMReply:
        self.seen_messages.append(list(messages))
        item = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _tool_reply(name: str, args: dict, call_id: str = "c1") -> LLMReply:
    return LLMReply(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args, ensure_ascii=False))],
        model="scripted",
    )


def _answer_reply(text: str) -> LLMReply:
    return LLMReply(content=text, tool_calls=[], model="scripted")


def _agent(llm: ScriptedLLM, max_steps: int = 5) -> ReactAgent:
    return ReactAgent(
        llm=llm,
        registry=build_default_registry(),
        settings=Settings(agent_max_steps=max_steps, llm_provider="mock"),
    )


# ============================================================================
# 终止条件 ①：拿到最终答案
# ============================================================================

def test_stops_when_model_answers_directly():
    llm = ScriptedLLM([_answer_reply("你好，有什么可以帮你？")])
    run = asyncio.run(_agent(llm).run("你好"))

    assert run.stopped_by == "answer"
    assert run.answer == "你好，有什么可以帮你？"
    assert run.trace == []
    assert run.steps == 0
    assert llm.calls == 1


def test_tool_then_answer():
    """典型两步：先调工具，拿到结果后给出答案。"""
    llm = ScriptedLLM([
        _tool_reply("calculator", {"expression": "123*456"}),
        _answer_reply("结果是 56088"),
    ])
    run = asyncio.run(_agent(llm).run("123*456 等于几"))

    assert run.stopped_by == "answer"
    assert run.steps == 1
    assert len(run.trace) == 1
    assert run.trace[0].tool == "calculator"
    assert run.trace[0].ok is True
    assert "56088" in run.trace[0].result
    assert llm.calls == 2


# ============================================================================
# 终止条件 ②：最大步数
# ============================================================================

def test_stops_at_max_steps():
    """⭐ 不加这个上限，模型陷入循环就会**无限烧钱**、请求永不返回。

    这里让模型每次都用不同参数调工具，绕过重复检测，
    从而精确测到步数上限这条分支。
    """
    replies = [_tool_reply("calculator", {"expression": f"{i}+1"}) for i in range(20)]
    llm = ScriptedLLM(replies)
    run = asyncio.run(_agent(llm, max_steps=3).run("算点东西"))

    assert run.stopped_by == "max_steps"
    assert run.steps == 3
    assert len(run.trace) == 3
    assert run.degraded is True
    assert "3 步" in run.answer


# ============================================================================
# 终止条件 ③：重复动作检测
# ============================================================================

def test_stops_on_repeated_action():
    """⭐ 加分项：连续两轮同样的工具 + 同样的参数 → 判定卡死，主动退出。

    有了它，就不必白烧掉 max_steps 次 token 才停下来。
    """
    same = {"expression": "1+1"}
    llm = ScriptedLLM([
        _tool_reply("calculator", same, call_id="c1"),
        _tool_reply("calculator", same, call_id="c2"),  # 完全重复
    ])
    run = asyncio.run(_agent(llm, max_steps=8).run("1+1"))

    assert run.stopped_by == "repeat"
    assert run.steps == 1  # 第 2 步就被拦下，没有执行
    assert len(run.trace) == 1
    assert run.degraded is True
    assert "重复" in run.answer


def test_repeat_detection_ignores_key_order():
    """参数 JSON 的 key 顺序不同，不算「不同的调用」。

    不做规范化的话，`{"a":1,"b":2}` 和 `{"b":2,"a":1}` 会被误判为两次不同调用，
    死循环检测就失效了。
    """
    llm = ScriptedLLM([
        _tool_reply("search_docs", {"query": "RAG", "top_k": 3}, call_id="c1"),
        _tool_reply("search_docs", {"top_k": 3, "query": "RAG"}, call_id="c2"),
    ])
    run = asyncio.run(_agent(llm, max_steps=8).run("查 RAG"))

    assert run.stopped_by == "repeat"


def test_different_args_not_treated_as_repeat():
    """参数变了就不算重复，循环要继续。"""
    llm = ScriptedLLM([
        _tool_reply("calculator", {"expression": "1+1"}, call_id="c1"),
        _tool_reply("calculator", {"expression": "2+2"}, call_id="c2"),
        _answer_reply("算完了"),
    ])
    run = asyncio.run(_agent(llm).run("算两次"))

    assert run.stopped_by == "answer"
    assert run.steps == 2
    assert len(run.trace) == 2


# ============================================================================
# 异常处理
# ============================================================================

def test_model_failure_degrades_gracefully():
    """模型挂了要返回友好错误，不能把异常抛给调用方。"""
    llm = ScriptedLLM([LLMError("接口 500", retryable=True)])
    run = asyncio.run(_agent(llm).run("你好"))

    assert run.stopped_by == "error"
    assert run.degraded is True
    assert "模型调用失败" in run.answer


def test_tool_failure_is_fed_back_not_raised():
    """⭐ 工具失败必须回填给模型，让它自己决定下一步。

    这是 Day 4 的设计核心：工具出错是正常业务流，不是服务器故障。
    """
    llm = ScriptedLLM([
        _tool_reply("calculator", {"expression": "非法((("}),
        _answer_reply("算不出来，表达式有问题"),
    ])
    run = asyncio.run(_agent(llm).run("算一下"))

    assert run.stopped_by == "answer"  # 循环没崩，正常走到答案
    assert run.trace[0].ok is False
    assert run.trace[0].result  # 错误信息被记进 trace 了
    # 错误信息确实被喂回了模型
    last_messages = llm.seen_messages[-1]
    tool_msgs = [m for m in last_messages if m.get("role") == "tool"]
    assert tool_msgs
    assert "失败" in tool_msgs[0]["content"]


def test_unknown_tool_does_not_crash():
    llm = ScriptedLLM([
        _tool_reply("不存在的工具", {}),
        _answer_reply("换个方式回答"),
    ])
    run = asyncio.run(_agent(llm).run("试试"))

    assert run.stopped_by == "answer"
    assert run.trace[0].ok is False
    assert "未注册" in run.trace[0].result


# ============================================================================
# 上下文与轨迹
# ============================================================================

def test_system_prompt_is_first_message():
    llm = ScriptedLLM([_answer_reply("好的")])
    asyncio.run(_agent(llm).run("问题"))

    first = llm.seen_messages[0][0]
    assert first["role"] == "system"
    assert "工具" in first["content"]


def test_history_is_included():
    llm = ScriptedLLM([_answer_reply("好的")])
    history = [
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]
    asyncio.run(_agent(llm).run("这一轮", history=history))

    roles = [m["role"] for m in llm.seen_messages[0]]
    assert roles == ["system", "user", "assistant", "user"]


def test_tool_result_appended_to_context():
    """工具结果必须以 role=tool 追加，否则模型看不到执行结果。"""
    llm = ScriptedLLM([
        _tool_reply("get_current_time", {"timezone": "Asia/Shanghai"}),
        _answer_reply("现在是……"),
    ])
    asyncio.run(_agent(llm).run("现在几点"))

    second_call = llm.seen_messages[1]
    roles = [m["role"] for m in second_call]
    assert "assistant" in roles
    assert "tool" in roles


def test_trace_records_duration():
    llm = ScriptedLLM([
        _tool_reply("calculator", {"expression": "1+1"}),
        _answer_reply("2"),
    ])
    run = asyncio.run(_agent(llm).run("1+1"))

    step = run.trace[0]
    assert step.step == 1
    assert step.duration_ms >= 0
    assert step.arguments
