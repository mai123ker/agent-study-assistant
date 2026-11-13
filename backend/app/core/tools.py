"""工具系统（Day 4 Function Calling + Day 20 权限与安全）。

模型不会"执行"任何东西 —— 它只是**返回一个 JSON**，告诉你它想调哪个函数、
传什么参数。真正的执行在我们这边。这个认知是理解 Function Calling 的起点。

本模块解决三件事：
1. 工具怎么声明给模型看（JSON Schema）
2. 模型选完工具后怎么执行（分发）
3. 执行前怎么保证安全（白名单 + 参数校验 + 路径穿越拦截）

面试要点（Day 20 核心）：
- 模型怎么"选"工具？没有魔法，就是靠 `description` 写得好不好。
  description 写烂 → 该调不调、或调错工具。
- 工具报错怎么办？**绝不能抛异常炸掉服务**。把错误格式化成文本回填给模型，
  让它自己决定下一步（换个工具 / 换个参数 / 告诉用户做不到）。
- 为什么不能靠"在 prompt 里说不要被注入"来防御？
  因为模型分不清"系统指令"和"文档内容" —— 这是原理层面的根因，
  只能靠代码层的白名单和参数校验兜底。
"""

from __future__ import annotations

import ast
import json
import logging
import operator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ============================================================================
# 工具执行结果
# ============================================================================

@dataclass
class ToolResult:
    """统一的工具返回结构。ok=False 时 error 一定有值。"""

    ok: bool
    content: str
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_tool_message(self, tool_call_id: str) -> dict[str, str]:
        """转成 OpenAI 协议里 role=tool 的那条消息。"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": self.content if self.ok else f"[工具执行失败] {self.error}",
        }


# ============================================================================
# 具体工具实现
# ============================================================================

# 只允许四则运算与取模，禁止任何属性访问 / 函数调用 / 下标
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# 幂运算上限：防止 9**9**9 这种把进程算死（DoS）
_MAX_POW_EXPONENT = 100


def _eval_ast(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        left, right = _eval_ast(node.left), _eval_ast(node.right)
        if op is operator.pow and isinstance(right, (int, float)) and right > _MAX_POW_EXPONENT:
            raise ValueError(f"指数过大（上限 {_MAX_POW_EXPONENT}）")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(_eval_ast(node.operand))
    # 任何 Attribute / Call / Subscript / Name 都在这里被拒
    raise ValueError(f"表达式含不允许的语法: {type(node).__name__}")


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（{timezone}）"


def calculator(expression: str) -> str:
    """安全的四则运算。用 AST 白名单而不是 eval()。

    面试加分点：直接 `eval()` 能被 `__import__('os').system(...)` 打穿，
    这是最典型的注入面。用 ast.parse + 白名单遍历可以彻底堵死。
    """
    tree = ast.parse(expression, mode="eval")
    value = _eval_ast(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expression} = {value}"


# 文档检索工具的真正实现由 rag 模块注入（避免循环 import）
_SEARCH_HOOK: Callable[[str, int], str] | None = None


def set_search_hook(fn: Callable[[str, int], str] | None) -> None:
    global _SEARCH_HOOK
    _SEARCH_HOOK = fn


def search_docs(query: str, top_k: int = 3) -> str:
    if _SEARCH_HOOK is None:
        raise RuntimeError("文档检索未就绪：尚未建立索引，或未注入检索钩子")
    return _SEARCH_HOOK(query, top_k)


# ---- 文件读取：必须限制在允许目录内（Day 20 路径穿越拦截）----

_ALLOWED_DIR: Path = (Path(__file__).resolve().parents[2] / "data" / "docs").resolve()


def set_allowed_dir(path: str | Path) -> None:
    global _ALLOWED_DIR
    _ALLOWED_DIR = Path(path).resolve()


def get_allowed_dir() -> Path:
    return _ALLOWED_DIR


def read_file(path: str) -> str:
    """读取允许目录内的文本文件。

    安全要点：用 `resolve()` 把 `..` 和软链接都展开后再比对，
    不能只做字符串前缀判断（`data/docs_evil` 会骗过 startswith）。
    """
    base = _ALLOWED_DIR
    target = (base / path).resolve()

    if not target.is_relative_to(base):
        raise PermissionError(
            f"路径越界：只允许读取 {base} 目录内的文件（收到: {path}）"
        )
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    return target.read_text(encoding="utf-8", errors="replace")


# ============================================================================
# 工具参数模型（Pydantic 校验，Day 20）
# ============================================================================

class EmptyArgs(BaseModel):
    """无参数工具。用空模型而不是 None，校验路径才统一。"""


class TimeArgs(BaseModel):
    timezone: str = Field(default="Asia/Shanghai", description="时区名")


class CalculatorArgs(BaseModel):
    expression: str = Field(..., min_length=1, max_length=200, description="数学表达式")


class SearchDocsArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="检索关键词")
    top_k: int = Field(default=3, ge=1, le=10, description="返回条数")


class ReadFileArgs(BaseModel):
    path: str = Field(..., min_length=1, max_length=300, description="相对允许目录的路径")


# ============================================================================
# 注册表
# ============================================================================

@dataclass
class ToolSpec:
    name: str
    description: str
    arg_model: type[BaseModel]
    func: Callable[..., str]
    dangerous: bool = False

    def json_schema(self) -> dict[str, Any]:
        """转成 OpenAI tools 参数要求的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arg_model.model_json_schema(),
            },
        }


class ToolRegistry:
    """工具白名单 + 分发器。

    白名单的意义：模型返回的工具名是不可信输入。
    如果直接 `getattr(module, name)()`，模型（或注入攻击）就能调到任意函数。
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._call_log: list[dict[str, Any]] = []

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.json_schema() for spec in self._tools.values()]

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """调用历史 —— Day 21 可观测性的雏形。"""
        return list(self._call_log)

    def execute(self, name: str, arguments: str | dict[str, Any]) -> ToolResult:
        """执行工具。**任何异常都在这里被兜住，绝不外抛。**"""
        started = datetime.now()

        def _log(ok: bool, error: str | None) -> None:
            self._call_log.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "ok": ok,
                    "error": error,
                    "duration_ms": int((datetime.now() - started).total_seconds() * 1000),
                    "at": started.isoformat(timespec="seconds"),
                }
            )

        # ① 白名单
        spec = self._tools.get(name)
        if spec is None:
            msg = f"未注册的工具 `{name}`。可用工具：{', '.join(self.names())}"
            logger.warning("工具调用被拒: %s", msg)
            _log(False, msg)
            return ToolResult(ok=False, content="", error=msg)

        # ② 参数解析 + 校验
        try:
            raw = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(raw, dict):
                raise ValueError("参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            msg = f"参数不是合法 JSON: {exc}"
            _log(False, msg)
            return ToolResult(ok=False, content="", error=msg)

        try:
            args = spec.arg_model(**raw)
        except ValidationError as exc:
            msg = f"参数校验失败: {exc.errors()[0].get('msg', '未知错误')}"
            _log(False, msg)
            return ToolResult(ok=False, content="", error=msg)

        # ③ 执行（兜住一切异常）
        try:
            output = spec.func(**args.model_dump())
        except PermissionError as exc:
            logger.warning("工具权限拦截: %s", exc)
            _log(False, str(exc))
            return ToolResult(ok=False, content="", error=str(exc))
        except Exception as exc:  # noqa: BLE001 — 这里就是要兜住全部
            logger.exception("工具执行异常: %s", name)
            _log(False, f"{type(exc).__name__}: {exc}")
            return ToolResult(ok=False, content="", error=f"{type(exc).__name__}: {exc}")

        _log(True, None)
        return ToolResult(ok=True, content=str(output))


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="get_current_time",
            description="获取当前日期和时间。当用户询问现在几点、今天日期、当前时间时必须调用。",
            arg_model=TimeArgs,
            func=get_current_time,
        )
    )
    reg.register(
        ToolSpec(
            name="calculator",
            description="计算数学表达式，支持加减乘除、整除、取模和乘方。当用户需要精确算术运算（尤其是数字较大或需要精确结果）时必须调用，不要自己心算。",
            arg_model=CalculatorArgs,
            func=calculator,
        )
    )
    reg.register(
        ToolSpec(
            name="search_docs",
            description="在本地资料库中检索相关文档片段。当用户询问的资料可能来自已有文档、笔记或 PDF 时调用。",
            arg_model=SearchDocsArgs,
            func=search_docs,
        )
    )
    reg.register(
        ToolSpec(
            name="read_file",
            description="读取本地资料目录中的某个文本文件。仅当明确知道文件名时使用。",
            arg_model=ReadFileArgs,
            func=read_file,
            dangerous=True,
        )
    )
    return reg
