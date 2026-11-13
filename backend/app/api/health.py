"""健康检查接口。

Day 1 它只回答「服务起来了没」。现在它还要回答运维最关心的两个问题：
- 模型走的是 mock 还是真实接口？（配置生效了吗）
- RAG 索引建好了没？（功能可用吗）

面试点：健康检查不该只返回 200。**能让调用方判断「我能不能用」**才有价值。
"""

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.core.rag import RagService
from app.deps import get_rag_service
from app.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(rag: RagService = Depends(get_rag_service)) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        rag_indexed=rag.is_indexed,
    )
