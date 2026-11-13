"""FastAPI 依赖注入。

把各个「有状态的重对象」做成单例依赖，路由里只声明参数、不自己 new。
这样以后换实现（换向量库、换 embedding、加缓存）不用动路由代码。

注意 `get_rag_service` 会顺手完成两件事：
1. 把 `tool_allow_dir` 设给工具层（Day 20 的路径白名单）
2. 把 RAG 检索注册到 `search_docs` 工具的钩子上（避免 tools ↔ rag 循环 import）
"""

from functools import lru_cache

from app.config import get_settings
from app.core import tools
from app.core.llm_client import LLMClient
from app.core.memory import ConversationMemory
from app.core.rag import RagService, register_service
from app.core.react_agent import ReactAgent
from app.core.tools import ToolRegistry, build_default_registry


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


@lru_cache
def get_memory() -> ConversationMemory:
    settings = get_settings()
    return ConversationMemory(max_turns=settings.memory_max_turns)


@lru_cache
def get_tool_registry() -> ToolRegistry:
    settings = get_settings()
    # Day 20：read_file 只允许读这个目录
    tools.set_allowed_dir(settings.allow_path)
    return build_default_registry()


@lru_cache
def get_rag_service() -> RagService:
    service = RagService(llm=get_llm_client())
    register_service(service)  # 接上 search_docs 工具的钩子
    return service


@lru_cache
def get_react_agent() -> ReactAgent:
    return ReactAgent(llm=get_llm_client(), registry=get_tool_registry())
