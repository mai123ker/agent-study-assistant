# Agent 求职学习助手

一个可演示的 **RAG + Agent** 项目。目标不是"能跑"，而是能在面试里讲清楚：
检索失败怎么处理、工具报错怎么兜底、循环怎么终止、上下文怎么膨胀。

> **当前状态**：Day 2 / 4 / 5 / 8–11 / 18 / 19 / 20 已落地，**152 个测试全绿**，
> 默认离线可跑（mock 模型 + hash 向量），不需要任何 API key。

## 文档导航

| 想做这件事 | 看这份 |
|---|---|
| **今天具体敲什么、改哪个文件、怎么验收** | `docs/逐步学习手册.md` |
| 一共几个知识点、优先级 | `docs/知识点清单.md` |

---

## 一、快速开始

### 方式 A：Docker（一条命令）

```bash
cd backend
cp .env.example .env      # 首次需要
docker compose up --build
```

打开 <http://127.0.0.1:8000/docs>。

### 方式 B：本地

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

打开 <http://localhost:5173>。

### 验证

```bash
cd backend
./.venv/Scripts/python.exe -m pytest -q
# 152 passed
```

---

## 二、三种对话模式

前端左上角可以切换，或直接调接口：

```bash
# 普通多轮对话
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好","mode":"chat"}'

# 资料问答（返回引用来源）
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"什么是 RAG","mode":"rag"}'

# Agent（ReAct 循环，可调工具）
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"123 * 456 等于多少","mode":"agent"}'
```

| mode | 链路 | 返回里值得看的字段 |
|---|---|---|
| `chat` | 记忆 → 模型 | `session_id` |
| `rag` | 记忆 → 检索 → 带引用生成 | `sources`（引用来源 + 相关度） |
| `agent` | ReAct 循环 | `trace`（每步工具调用）、`stopped_by`（终止原因） |

### 检索对比实验（Day 11 那张表的原始数据）

```bash
curl -X POST http://127.0.0.1:8000/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query":"MCP 协议","top_k":3,"mode":"vector"}'
# 把 mode 换成 bm25 / hybrid 各跑一遍，对比 hits 里的 vector_score / bm25_score / fused_score
```

---

## 三、目录结构

```text
agent-study-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # 入口 + CORS + 启动时建索引
│   │   ├── config.py            # 集中配置（全部走 pydantic-settings）
│   │   ├── schemas.py           # 接口契约
│   │   ├── deps.py              # 依赖注入（单例）
│   │   ├── api/
│   │   │   ├── health.py        # GET  /health
│   │   │   ├── chat.py          # POST /api/chat · /api/chat/stream
│   │   │   ├── rag.py           # POST /api/rag/index · /api/rag/search
│   │   │   └── tools.py         # GET  /api/tools · POST /api/tools/{name}/call
│   │   └── core/
│   │       ├── llm_client.py    # 模型调用（多轮 / 工具 / 流式 / 重试）
│   │       ├── memory.py        # 会话记忆 + 滑动窗口      [Day 2]
│   │       ├── tools.py         # 工具注册 + 权限校验      [Day 4/20]
│   │       ├── react_agent.py   # 手写 ReAct 循环          [Day 5]
│   │       ├── loader.py        # PDF/TXT 解析            [Day 8]
│   │       ├── splitter.py      # 三种切块策略            [Day 9]
│   │       ├── embeddings.py    # 三种 Embedding 提供方   [Day 10]
│   │       ├── vector_store.py  # 内存 / Chroma 双后端    [Day 10]
│   │       ├── bm25.py          # 纯 Python BM25          [Day 11]
│   │       ├── retriever.py     # 混合检索 + RRF 融合     [Day 11]
│   │       └── rag.py           # RAG 链路组装
│   ├── tests/                   # 152 个测试，全部离线可跑
│   ├── data/docs/               # 放你的资料（已带 2 份示例）
│   ├── Dockerfile
│   └── docker-compose.yml
└── frontend/
    └── src/
        ├── App.vue              # 模式切换 / 流式渲染 / 来源与轨迹展示
        └── api.js               # 接口层（含 SSE 解析）
```

---

## 四、请求流程

```text
浏览器输入框
   │  v-model 绑定 input + 模式选择
   ▼
App.vue handleSend()
   │  流式 → streamMessage()   │  非流式 → sendMessage()
   ▼
POST /api/chat/stream (SSE)    或    POST /api/chat
   │  Vite 代理转发到 127.0.0.1:8000
   ▼
FastAPI 路由 app/api/chat.py
   │  Pydantic 校验 ChatRequest
   ▼
按 mode 分流 ──┬── chat  → ConversationMemory → LLMClient
              ├── rag   → RagService.answer() → 检索 → 拼 prompt → LLMClient
              └── agent → ReactAgent.run() → 循环 { LLMClient → execute_tool }
   ▼
ChatResponse / SSE 事件流
   │
   ▼
前端渲染气泡 + 引用来源 + 工具轨迹
```

---

## 五、技术选型说明（面试会被问）

| 选择 | 为什么 | 局限 / 什么时候换 |
|---|---|---|
| 向量库默认用**纯 Python 内存实现** | 零依赖，5 分钟跑通全流程 | 上万条后暴力检索变慢，切 `vector_backend=chroma` |
| BM25 **自己实现**，用 CJK 二元组分词 | 公式只有 20 行，手写过才答得上 k1/b 的含义 | 不如 jieba 准，但零依赖 |
| 混合检索用 **RRF** 而不是加权求和 | 余弦在 0~1，BM25 无上界，加权会让 BM25 一边倒 | RRF 丢弃了分数信息，数据量大时可考虑归一化加权 |
| Embedding 默认 **hash**（仅测试） | 让测试离线、确定性 | **没有语义能力**，真实使用要切 `api` 或 `local` |
| 会话记忆用 **deque(maxlen)** | 淘汰 O(1)，不用自己写 pop(0) | 单进程内存，重启即丢；生产要换 Redis |
| 工具用 **ast 白名单**而不是 eval | eval 能被 `__import__('os')` 打穿 | 只支持四则运算，够用 |

---

## 六、常见卡点

| 现象 | 原因 | 解法 |
|---|---|---|
| `uvicorn: command not found` | 虚拟环境没激活 | 命令行前应出现 `(.venv)` |
| 端口被占用 | 8000 有别的程序 | 改 `--port 8001`，同步改 `vite.config.js` 的 target |
| 浏览器报 CORS | 代理没配 / 后端没加中间件 | 两处都要有 |
| 改了 `.env` 没生效 | 配置在启动时读取 | **重启 uvicorn** |
| `npm install` 卡住 | 默认源慢 | `npm config set registry https://registry.npmmirror.com` |
| 检索结果全是无关内容 | `embedding_provider=hash` 不懂语义 | 改成 `api`（配 LLM key 即可）或 `local` |
| `search_docs` 报"未就绪" | 索引没建 | `POST /api/rag/index` |
| `tool_calls` 解析不出来 | 结构没看清 | `print(json.dumps(..., indent=2))` 先看原始数据 |
| 流式前端不渲染 | 响应头不对 / 被 Nginx 缓冲 | 检查 `text/event-stream` 和 `X-Accel-Buffering` |

---

## 七、Roadmap

- **Day 3**：结构化输出 + JSON 解析兜底（`chat_json` 已实现，接口未接）
- **Day 6**：Reflection / Plan-and-Solve
- **Day 7**：Transformer 面试 10 问
- **Day 12**：Rerank（Cross-Encoder 精排）
- **Day 13**：上下文裁剪策略 + 长期记忆
- **Day 14**：LangGraph 重写（**手写版不要删**，面试时对比着讲）
- **Day 15–17**：MCP Server
- **Day 21**：结构化日志 + trace_id（`ToolRegistry.call_log` 是雏形）
- **Day 22–25**：评测集 + 优化对比表 ← **面试杀手锏**
- **Day 26–30**：README 收尾、复盘讲稿、简历、投递

---

## 八、接真实模型

推荐阿里云百炼：每个模型 100 万 Token 免费，一个 key 调多家模型。

```ini
# backend/.env
LLM_PROVIDER=openai
LLM_API_KEY=sk-你的key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

# 想要真实语义检索（强烈建议一起改）
EMBEDDING_PROVIDER=api
```

改完**必须重启** —— 配置是启动时读取的。
