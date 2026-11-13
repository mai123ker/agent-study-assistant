"""FastAPI 入口。

启动：uvicorn app.main:app --reload --port 8000
文档：http://127.0.0.1:8000/docs

启动时会尝试自动建一次 RAG 索引（文档目录为空就跳过）。
索引失败**不阻塞启动** —— 一个可选功能不该让整个服务起不来，
这是 Day 19「降级」思想在启动流程上的体现。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health, rag as rag_api, tools as tools_api
from app.config import get_settings
from app.core.rag import RagService
from app.deps import get_rag_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭钩子。

    在这里做「索引预热」：如果文档目录里有东西且还没索引，就建一次。
    这样演示时不用先手动调接口。
    """
    service: RagService = get_rag_service()
    docs_dir = settings.docs_path

    if not docs_dir.exists():
        logger.warning("文档目录不存在，RAG 功能不可用: %s", docs_dir)
    elif not any(docs_dir.rglob("*")):
        logger.warning("文档目录为空，RAG 检索会返回空结果: %s", docs_dir)
    else:
        try:
            stats = await service.index()
            logger.info(
                "RAG 索引就绪：%d 个文件 / %d 个片段（embedding=%s, backend=%s）",
                stats.indexed_files, stats.total_chunks,
                service.embedder.name, service.store.backend,
            )
        except Exception:  # noqa: BLE001 — 索引失败不该阻塞服务启动
            logger.exception("启动时建索引失败，服务继续启动。可稍后手动 POST /api/rag/index")

    yield

    logger.info("服务关闭")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Agent 求职学习助手 —— 从最小闭环到可演示的 RAG + Agent 项目。\n\n"
        "三种对话模式：`chat` 普通多轮 / `rag` 基于资料回答 / `agent` 带工具调用。"
    ),
    lifespan=lifespan,
)

# 前后端分离，浏览器会跨域；放开本地前端端口
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(rag_api.router)
app.include_router(tools_api.router)


@app.get("/", summary="根路径")
async def root() -> dict:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "modes": ["chat", "rag", "agent"],
    }
