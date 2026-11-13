"""LLM 客户端测试（Day 3 结构化输出 + Day 19 错误分类）。

错误分类（哪些该重试、哪些不该）是「有没有在生产里踩过」的分水岭，
所以单独测它。
"""

import pytest

from app.config import Settings
from app.core.llm_client import LLMClient


# ============================================================================
# JSON 解析兜底（Day 3）
# ============================================================================

def test_parse_clean_json():
    parsed, err = LLMClient._parse_json('{"intent": "提问", "confidence": 0.9}')
    assert err is None
    assert parsed["intent"] == "提问"


def test_parse_json_in_code_fence():
    """模型很爱把 JSON 包在 ```json ``` 里 —— 这是最常见的坏法。"""
    text = '好的，结果如下：\n```json\n{"a": 1}\n```\n希望有帮助。'
    parsed, err = LLMClient._parse_json(text)
    assert err is None
    assert parsed == {"a": 1}


def test_parse_json_with_surrounding_prose():
    text = '好的，这是结果：{"intent": "查询", "confidence": 0.8} 以上。'
    parsed, err = LLMClient._parse_json(text)
    assert err is None
    assert parsed["intent"] == "查询"


def test_parse_json_plain_fence():
    parsed, err = LLMClient._parse_json('```\n{"x": 2}\n```')
    assert err is None
    assert parsed == {"x": 2}


@pytest.mark.parametrize("bad", ["", "   ", "这不是 JSON", "{坏掉的 json", "null", "[1,2]"])
def test_parse_json_failure_returns_error(bad):
    """解析失败必须返回 (None, 错误)，**不能抛异常**。

    否则一次模型抽风就会把整条链路打挂。
    """
    parsed, err = LLMClient._parse_json(bad)
    assert parsed is None
    assert err


def test_parse_json_rejects_non_object():
    """顶层是数组也不算合格 —— 契约要求是对象。"""
    parsed, err = LLMClient._parse_json('[{"a": 1}]')
    assert parsed is None
    assert err


# ============================================================================
# 错误分类（Day 19）
# ============================================================================

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_status_codes(status):
    """限流和服务端问题：重试有意义。"""
    err = LLMClient._classify_http_error(status, "body")
    assert err.retryable is True
    assert err.status_code == status


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_status_codes(status):
    """参数错、key 错、无权限、路径错：重试一万次也一样，直接失败。"""
    err = LLMClient._classify_http_error(status, "body")
    assert err.retryable is False


def test_error_message_truncates_body():
    """错误体可能很长，日志里要截断。"""
    err = LLMClient._classify_http_error(500, "x" * 5000)
    assert len(str(err)) < 500


# ============================================================================
# 指数退避（Day 19）
# ============================================================================

def test_backoff_grows_exponentially():
    """对方限流时立刻重试只会加剧限流 —— 必须退避。"""
    client = LLMClient(Settings(llm_provider="mock", llm_retry_base_delay=0.5))

    d1 = client._backoff_delay(1)
    d2 = client._backoff_delay(2)
    d3 = client._backoff_delay(3)

    assert d1 == 0.5
    assert d2 == 1.0
    assert d3 == 2.0
    assert d1 < d2 < d3


def test_backoff_is_capped():
    """退避要有上限，否则重试间隔会长到不可接受。"""
    client = LLMClient(Settings(llm_provider="mock", llm_retry_base_delay=1, llm_retry_max_delay=4))
    assert client._backoff_delay(10) == 4


# ============================================================================
# mock 行为
# ============================================================================

def _mock_client() -> LLMClient:
    return LLMClient(Settings(llm_provider="mock"))


def test_mock_picks_calculator():
    name, args = LLMClient._pick_mock_tool("123 * 456 等于多少", {"calculator"})
    assert name == "calculator"
    assert args["expression"] == "123 * 456"


def test_mock_picks_time_tool():
    name, _ = LLMClient._pick_mock_tool("现在几点了", {"get_current_time"})
    assert name == "get_current_time"


def test_mock_does_not_pick_tool_for_chitchat():
    """闲聊不该调工具 —— mock 也要遵守这个行为，否则测不出真实问题。"""
    name, _ = LLMClient._pick_mock_tool("你好呀", {"calculator", "get_current_time", "search_docs"})
    assert name is None


def test_mock_skips_tool_not_available():
    name, _ = LLMClient._pick_mock_tool("现在几点了", {"calculator"})
    assert name is None


def test_mock_returns_answer_after_tool_result():
    """有 tool 消息后必须收口给答案，否则 ReAct 循环会停不下来。"""
    messages = [
        {"role": "user", "content": "1+1"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "1+1 = 2"},
    ]
    reply = _mock_client()._mock_chat(messages, [{"type": "function", "function": {"name": "calculator"}}])
    assert reply.tool_calls == []
    assert "2" in reply.content


def test_mock_rag_prompt_produces_answer_not_echo():
    """RAG 提示词不该被原样回显（演示时很难看）。"""
    prompt = (
        "你是资料助手……\n===== 资料开始 =====\n"
        "[1] （来源：示例.md）\nRAG 是检索增强生成。\n"
        "===== 资料结束 =====\n\n用户问题：RAG 是什么\n"
    )
    reply = _mock_client()._mock_chat([{"role": "user", "content": prompt}], None)
    assert "资料开始" not in reply.content
    assert "RAG 是什么" in reply.content


# ============================================================================
# 配置
# ============================================================================

def test_mock_mode_when_no_key():
    assert LLMClient(Settings(llm_provider="openai", llm_api_key="")).is_mock is True
    assert LLMClient(Settings(llm_provider="openai", llm_api_key="sk-x")).is_mock is False


def test_model_name_reflects_mode():
    assert LLMClient(Settings(llm_provider="mock")).model_name == "mock"
    assert LLMClient(Settings(llm_provider="openai", llm_api_key="sk-x", llm_model="qwen-plus")).model_name == "qwen-plus"
