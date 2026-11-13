"""工具系统测试（Day 4 + Day 20）。

重点在**安全**：白名单、参数校验、路径穿越、代码注入。
这些是面试时最有说服力的一段 —— 因为大多数候选人根本没想到要测这些。
"""

import json

import pytest

from app.core import tools
from app.core.tools import ToolRegistry, build_default_registry


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry()


# ============================================================================
# 注册与白名单
# ============================================================================

def test_default_tools_registered(registry):
    assert set(registry.names()) == {
        "get_current_time",
        "calculator",
        "search_docs",
        "read_file",
    }


def test_schemas_are_openai_compatible(registry):
    for schema in registry.schemas():
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_unknown_tool_rejected(registry):
    """⭐ 白名单的意义：模型返回的工具名是**不可信输入**。

    如果直接 getattr(module, name)()，模型（或注入攻击）就能调到任意函数。
    """
    result = registry.execute("os_system", {"cmd": "rm -rf /"})
    assert result.ok is False
    assert "未注册" in result.error


# ============================================================================
# 参数校验
# ============================================================================

def test_bad_json_arguments(registry):
    result = registry.execute("calculator", "{不是合法 json")
    assert result.ok is False
    assert "合法 JSON" in result.error


def test_arguments_must_be_object(registry):
    result = registry.execute("calculator", "[1, 2, 3]")
    assert result.ok is False


def test_missing_required_argument(registry):
    result = registry.execute("calculator", {})
    assert result.ok is False
    assert "校验失败" in result.error


def test_out_of_range_argument(registry):
    """top_k 上限是 10，传 999 应该被 Pydantic 拦住。"""
    result = registry.execute("search_docs", {"query": "x", "top_k": 999})
    assert result.ok is False


# ============================================================================
# 计算器安全（用 ast 而不是 eval）
# ============================================================================

@pytest.mark.parametrize("expr,expected", [
    ("1+1", "2"),
    ("2*3+4", "10"),
    ("(1+2)*3", "9"),
    ("10/4", "2.5"),
    ("10//3", "3"),
    ("10%3", "1"),
    ("2**10", "1024"),
    ("-5+3", "-2"),
])
def test_calculator_arithmetic(registry, expr, expected):
    result = registry.execute("calculator", {"expression": expr})
    assert result.ok is True
    assert expected in result.content


@pytest.mark.parametrize("evil", [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__",
    "exec('x=1')",
    "[1,2,3][0]",
    "lambda: 1",
])
def test_calculator_blocks_injection(registry, evil):
    """⭐ 这些表达式用 eval() 全都能打穿，用 ast 白名单则全部拒绝。"""
    result = registry.execute("calculator", {"expression": evil})
    assert result.ok is False


def test_calculator_blocks_huge_power(registry):
    """防止 9**9**9 把进程算死（DoS）。"""
    result = registry.execute("calculator", {"expression": "9**999"})
    assert result.ok is False
    assert "指数过大" in result.error


# ============================================================================
# 路径穿越（Day 20 核心）
# ============================================================================

def test_read_file_within_allowed_dir(registry, tmp_path):
    tools.set_allowed_dir(tmp_path)
    (tmp_path / "note.txt").write_text("这是允许读的内容", encoding="utf-8")

    result = registry.execute("read_file", {"path": "note.txt"})
    assert result.ok is True
    assert "允许读的内容" in result.content


@pytest.mark.parametrize("evil", [
    "../../../etc/passwd",
    "../../.env",
    "..\\..\\Windows\\win.ini",
    "....//....//etc/passwd",
    "sub/../../../outside.txt",
])
def test_read_file_blocks_traversal(registry, tmp_path, evil):
    """⭐ 用 resolve() 展开 .. 和软链接后再比对。

    只做字符串 startswith 判断会被 `data/docs_evil` 这类路径骗过。
    """
    allowed = tmp_path / "docs"
    allowed.mkdir()
    tools.set_allowed_dir(allowed)

    result = registry.execute("read_file", {"path": evil})
    assert result.ok is False
    assert "越界" in result.error or "不存在" in result.error


def test_read_file_blocks_sibling_dir_prefix_bypass(registry, tmp_path):
    """`docs_evil` 和 `docs` 是同级目录，前缀相同但不是子目录。

    这是 startswith 写法的经典漏洞，用 is_relative_to 才能挡住。
    """
    allowed = tmp_path / "docs"
    evil = tmp_path / "docs_evil"
    allowed.mkdir()
    evil.mkdir()
    (evil / "secret.txt").write_text("不该被读到", encoding="utf-8")

    tools.set_allowed_dir(allowed)
    result = registry.execute("read_file", {"path": "../docs_evil/secret.txt"})

    assert result.ok is False
    assert "不该被读到" not in result.content


# ============================================================================
# 异常兜底
# ============================================================================

def test_tool_exception_is_caught(registry):
    """⭐ 工具失败在 Agent 里是**正常业务流**，不是服务器故障。

    任何异常都必须被兜住并转成文本，回填给模型让它自己决定下一步。
    """
    from app.core.tools import EmptyArgs, ToolSpec

    def boom() -> str:
        raise RuntimeError("模拟工具炸了")

    registry.register(ToolSpec(
        name="boom", description="必定失败", arg_model=EmptyArgs, func=boom
    ))

    result = registry.execute("boom", {})
    assert result.ok is False
    assert "RuntimeError" in result.error
    assert "模拟工具炸了" in result.error


def test_tool_message_format(registry):
    """失败的工具结果也要能转成 role=tool 消息，否则循环会断。"""
    from app.core.tools import ToolResult

    ok_msg = ToolResult(ok=True, content="结果").as_tool_message("call_1")
    assert ok_msg["role"] == "tool"
    assert ok_msg["tool_call_id"] == "call_1"
    assert ok_msg["content"] == "结果"

    bad_msg = ToolResult(ok=False, content="", error="炸了").as_tool_message("call_2")
    assert "工具执行失败" in bad_msg["content"]


# ============================================================================
# 调用日志（Day 21 雏形）
# ============================================================================

def test_call_log_records_everything(registry):
    registry.execute("calculator", {"expression": "1+1"})
    registry.execute("calculator", {"expression": "非法("})

    log = registry.call_log
    assert len(log) == 2
    assert log[0]["ok"] is True
    assert log[1]["ok"] is False
    assert all("duration_ms" in entry for entry in log)
