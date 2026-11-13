# 基础镜像选 slim 版：完整版 1GB+，slim 只要 ~150MB
FROM python:3.13-slim

# PYTHONUNBUFFERED：日志实时输出，否则 docker logs 要等缓冲满了才显示
# PYTHONDONTWRITEBYTECODE：容器里不需要 .pyc，少一层写入
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ⚠️ 顺序很关键：先 copy requirements.txt 再装依赖，最后才 copy 代码。
# 这样改代码不会让依赖层缓存失效 —— 重建从「几分钟」变成「几秒」。
# 反过来写（先 copy . 再 pip install），每次改一行代码都要重装全部依赖。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests

# 非 root 运行。容器逃逸时能少一分风险，面试可以提一句
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查直接用我们自己的 /health —— 它能反映「依赖是否就绪」
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
