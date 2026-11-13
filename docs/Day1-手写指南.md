# Day 1 手写指南：从零敲出 Vue → FastAPI → 大模型 最小闭环

> 用法：**开着这个文件，一行行敲。每步写完立刻运行验证，通过了再往下。**
> 卡住了先自己想 10 分钟，再去看 `agent-study-assistant/`（答案对照）。

---

## 开始前（10 分钟）

### 为什么手写而不是 clone 项目

面试官不会问你"跑通过什么项目"，他会问：

- 为什么聊天接口用 `POST` 不用 `GET`？
- `async def` 在这里解决什么问题？
- 前端跨域报错是怎么产生的？
- 模型挂了你的服务为什么不会崩？

**这些问题的答案，只有你自己一行行写出来才记得住。**

### 建文件夹

**不要用我已经建好的 `agent-study-assistant`**，那份当答案。你新建一个：

```powershell
cd E:\
mkdir my-agent-study
cd my-agent-study
```

### 确认环境

```powershell
python --version    # 要 3.10+；如果报错就试 py --version
node --version      # 要 18+
npm --version
```

> 你本机有 Python 3.13 和 Node，应该都没问题。

---

## 步骤 0：搭骨架（15 分钟）

### 建目录

```powershell
mkdir backend
mkdir backend\app
mkdir backend\app\api
mkdir backend\app\core
mkdir frontend
```

### 建虚拟环境

```powershell
cd backend
python -m venv .venv
```

**为什么要虚拟环境？** 让这个项目的依赖跟系统 Python 隔离。以后每个项目一个 `.venv`，
A 项目要 fastapi 0.100、B 项目要 0.115，才不会打架。

### 激活它

```powershell
.\.venv\Scripts\Activate.ps1
```

> 如果报"禁止运行脚本"，执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> 然后重新激活。
>
> CMD 用户用：`.venv\Scripts\activate.bat`

**激活成功的标志**：命令行前面多了 `(.venv)`。

### 写依赖文件

新建 `backend\requirements.txt`，敲进去：

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pydantic-settings==2.7.0
httpx==0.28.1
pytest==8.3.4
```

**每个是干嘛的**（面试可能问）：

| 包 | 作用 |
|---|---|
| `fastapi` | Web 框架，负责路由、参数校验 |
| `uvicorn` | ASGI 服务器，真正监听端口、处理 HTTP 的 |
| `pydantic` | 数据校验，把 JSON 变成有类型的 Python 对象 |
| `pydantic-settings` | 从 `.env` 读配置 |
| `httpx` | 异步 HTTP 客户端，用来调大模型接口 |
| `pytest` | 测试框架 |

### 安装

```powershell
pip install -r requirements.txt
```

**验收**：没报红就对了。如果下载慢：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 步骤 1：让服务器先活过来（20 分钟）

**先写最小版本**，别一上来就写全。目标：浏览器能访问到一个响应。

新建 `backend\app\main.py`：

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "服务已启动"}
```

新建 `backend\app\__init__.py`（空文件，告诉 Python 这是个包）。

### 跑起来

```powershell
uvicorn app.main:app --reload --port 8000
```

**这一行的意思**（面试常问）：

- `app.main` → 找 `app/main.py` 这个模块
- `:app` → 用里面名叫 `app` 的那个变量
- `--reload` → 改代码自动重启（只在开发时用）
- `--port 8000` → 监听 8000

### 验收

浏览器打开 **http://127.0.0.1:8000/docs**

你应该看到一个自动生成的接口文档页面，里面有 `GET /`。
点 "Try it out" → "Execute"，能看到 `{"message": "服务已启动"}`。

> **这一步过了，说明 FastAPI + uvicorn 这条链路是通的。** 很多人卡在这里就是因为
> 一上来写了一堆代码，报错也不知道是哪层的问题。

**概念**：`@app.get("/")` 是**装饰器**，意思是"当有人用 GET 方法访问 `/` 时，执行下面这个函数"。
`async def` 表示这是异步函数——后面调大模型时会解释为什么重要。

---

## 步骤 2：加一个健康检查（10 分钟）

在 `main.py` 里追加：

```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

保存后 uvicorn 会自动重启。刷新 `/docs`，多了一个 `GET /health`。

**为什么要有这个接口？** 部署后运维/监控系统靠它判断"服务是不是活着"。
面试问"你怎么做服务健康检查"，这就是答案。

**验收**：浏览器直接访问 http://127.0.0.1:8000/health → 看到 `{"status":"ok"}`

---

## 步骤 3：定义接口契约（20 分钟）

现在要加聊天接口了。**先定义"请求长什么样、响应长什么样"**，再写逻辑。

新建 `backend\app\schemas.py`：

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    reply: str
    model: str
    latency_ms: int
```

**Pydantic 在这里解决什么问题？**（高频面试题）

1. **自动校验**：前端传 `""` 或干脆不传 `message`，FastAPI 直接返回 422 错误，
   你的业务代码根本不会被调用——省掉一堆 `if not message: return error`
2. **自动文档**：这些模型会出现在 `/docs` 里
3. **类型安全**：`req.message` 一定是 `str`，IDE 能补全，不用猜

`Field(...)` 里的 `...` 表示**必填**；`min_length=1` 表示不能是空字符串。

---

## 步骤 4：写出 /api/chat（先返回假数据）（20 分钟）

**关键**：先返回假数据，把接口跑通，再去接真模型。这样出问题你知道是接口的锅还是模型的锅。

新建 `backend\app\api\__init__.py`（空文件）。

新建 `backend\app\api\chat.py`：

```python
from fastapi import APIRouter

from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return ChatResponse(
        reply=f"（假数据）我收到了：{req.message}",
        model="fake",
        latency_ms=0,
    )
```

在 `main.py` 里挂上去：

```python
from app.api import chat

app.include_router(chat.router)
```

**为什么要 `APIRouter`？** 接口一多，全塞 `main.py` 会爆炸。
`APIRouter` 让你按业务分文件，`prefix="/api"` 统一加前缀。
面试问"你怎么组织项目结构"，这就是答案。

**为什么聊天用 `POST` 不用 `GET`？**（必问）

- `GET` 的参数放在 URL 里，会出现在浏览器历史、服务器日志里——**问题内容会被记录**
- URL 有长度限制，长问题会超
- `GET` 语义上是"读取"，`POST` 才是"提交数据让服务器处理"
- 请求体是 JSON，`GET` 不好带 body

### 验收

刷新 `/docs` → 找到 `POST /api/chat` → Try it out → 填 `{"message": "你好"}` → Execute

**必须测这两个边界**：

1. 传 `{"message": ""}` → 应该返回 **422**（Pydantic 挡住了）
2. 传 `{}` → 也应该返回 **422**

> 422 是 FastAPI 校验失败的标准状态码。**能说出这个，面试官会知道你用过真的。**

---

## 步骤 5：把配置抽出去（20 分钟）

**别把 API Key 写在代码里。** 新建 `backend\app\config.py`：

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Agent Study Assistant"
    llm_provider: str = "mock"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    request_timeout: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

新建 `backend\.env`：

```text
LLM_PROVIDER=mock
LLM_API_KEY=
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

**为什么？**

- Key 写进代码 → 一旦提交 Git，全世界都能看到，可能被刷爆
- `.env` 加进 `.gitignore`，就不会被提交
- 换模型/换供应商只改配置，不动代码

新建 `backend\.gitignore`：

```text
.venv/
__pycache__/
.env
*.pyc
```

**验收**：`print(get_settings().app_name)` 能打出值。

---

## 步骤 6：真正调用大模型（40 分钟）

这是今天的核心。新建 `backend\app\core\__init__.py`（空文件）。

新建 `backend\app\core\llm_client.py`：

```python
import time

import httpx

from app.config import get_settings


class LLMClient:
    def __init__(self):
        self.settings = get_settings()

    async def chat(self, message: str) -> tuple[str, int]:
        start = time.perf_counter()
        if self.settings.llm_provider == "mock" or not self.settings.llm_api_key:
            reply = f"[MOCK] 收到你的问题：{message}"
        else:
            reply = await self._remote_chat(message)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return reply, latency_ms

    async def _remote_chat(self, message: str) -> str:
        url = f"{self.settings.llm_base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": message}],
        }
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
```

改 `api/chat.py`，接上它：

```python
from fastapi import APIRouter

from app.core.llm_client import LLMClient
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    client = LLMClient()
    reply, latency_ms = await client.chat(req.message)
    return ChatResponse(
        reply=reply,
        model=client.settings.llm_model,
        latency_ms=latency_ms,
    )
```

### 三个必须搞懂的点

**① `messages` 里的角色**

```python
"messages": [{"role": "user", "content": message}]
```

- `system` → 设定 AI 的身份和规则（"你是一个学习助手，不确定就说不确定"）
- `user` → 用户说的话
- `assistant` → AI 之前说过的话（多轮对话时把历史带上）

**② 为什么用 `async def`？**（高频面试题）

调大模型要等好几秒。如果用同步（`def`），这几秒里整个服务**只能干等**，
别的用户请求全部排队。用 `async`，等待期间服务器可以去处理别的请求。

> 面试话术：「模型调用是 I/O 密集型操作，`await` 期间事件循环会切去处理其他请求，
> 所以并发能力比同步阻塞高得多。」

**③ 为什么单独抽一个 `llm_client.py`？**

- 路由函数只负责 HTTP，不负责调模型——**职责分离**
- 以后加超时、重试、流式、换模型，只改这一个文件
- 面试问"你怎么做错误重试"，你能指出具体在哪一层做

### 验收

先不填 Key（走 mock）：

1. `/docs` 里调 `POST /api/chat` → 返回 `[MOCK] 收到你的问题：...`
2. **故意把 `llm_base_url` 改错**，填个假 Key → 服务应该返回 500 而不是整个崩掉
3. 看终端，应该打印出了请求日志

**第 2 条很重要**：这就是面试问的"模型调用失败怎么办"。今天你先看到它怎么失败的，
Day 2 我们再教你怎么优雅处理（超时、重试、降级）。

### 想接真模型

去 DeepSeek 官网申请 Key，改 `.env`：

```text
LLM_PROVIDER=openai
LLM_API_KEY=sk-你的key
```

> DeepSeek / 通义 / 本地 Ollama 都是 OpenAI 兼容接口，只改 `LLM_BASE_URL` 和 `LLM_MODEL`。

---

## 步骤 7：Vue 前端接上（45 分钟）

```powershell
cd E:\my-agent-study
npm create vite@latest frontend -- --template vue
cd frontend
npm install
```

### 配代理（解决跨域）

改 `frontend\vite.config.js`：

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  }
})
```

**跨域是怎么回事？**（必问）

前端跑在 `5173`，后端在 `8000`——**端口不同就算跨域**。
浏览器出于安全策略，默认禁止前端直接请求别的端口。

两种解法：
- **后端开 CORS**（后端在响应头里说"我允许 5173 访问我"）
- **前端开代理**（让 5173 假装自己转发，浏览器就看不到跨域了）

这里用代理。**但后端也得开 CORS**，因为生产环境前端可能部署在别的域名——
在 `main.py` 里加：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 改 App.vue

只做三个元素：**输入框、按钮、回答区**。别做聊天气泡、Markdown 渲染——今天不需要。

```vue
<script setup>
import { ref } from 'vue'

const input = ref('')
const answer = ref('')
const loading = ref(false)
const errorMsg = ref('')

async function handleSend() {
  const text = input.value.trim()
  if (!text || loading.value) return   // 空输入不发、防重复提交

  loading.value = true
  errorMsg.value = ''
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    })
    if (!res.ok) throw new Error(`请求失败 (${res.status})`)
    const data = await res.json()
    answer.value = data.reply
  } catch (e) {
    errorMsg.value = e.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div style="max-width: 600px; margin: 40px auto; font-family: sans-serif">
    <h2>Agent 学习助手</h2>

    <textarea v-model="input" rows="3" style="width: 100%"></textarea>

    <button :disabled="loading || !input.trim()" @click="handleSend">
      {{ loading ? '生成中…' : '发送' }}
    </button>

    <p v-if="errorMsg" style="color: red">{{ errorMsg }}</p>

    <div v-if="answer" style="margin-top: 16px; padding: 12px; background: #f1f3f5">
      {{ answer }}
    </div>
  </div>
</template>
```

### 这三个细节必须有（面试会看）

- `loading` → 请求中按钮禁用，防止重复提交
- `!input.trim()` → 空输入不发请求
- `errorMsg` → 后端挂了要提示用户，不能白屏

### 验收

两个终端分别跑：

```powershell
# 终端 1
cd E:\my-agent-study\backend
uvicorn app.main:app --reload --port 8000

# 终端 2
cd E:\my-agent-study\frontend
npm run dev
```

浏览器打开 http://localhost:5173 → 输入问题 → 点发送 → **看到回答**。

---

## 步骤 8：验收 + 第一次提交（20 分钟）

### 逐条打勾

- [ ] `/docs` 能打开
- [ ] `/health` 返回 `{"status":"ok"}`
- [ ] 空输入返回 422
- [ ] 模型调用失败不会让服务崩
- [ ] API Key 在 `.env` 里，没写死在代码里
- [ ] 前端能发出请求并显示回答
- [ ] 请求中按钮禁用

### 建仓库并提交

```powershell
cd E:\my-agent-study
git init
git add .
git commit -m "feat: complete minimal Vue FastAPI LLM chat flow"
```

**提交前一定要确认**：`git status` 里**不能出现 `.env`**。如果出现了，说明 `.gitignore` 没生效。

### 写 README

至少写清楚三件事：

1. 怎么启动（后端、前端分别什么命令）
2. 画一张调用链：`Vue → FastAPI → LLM API → FastAPI → Vue`
3. **今天踩的三个坑 + 怎么解决的**

> 第 3 条是面试金矿。面试官问"你遇到过什么困难"，
> "跨域报错排查了半天，最后发现是代理没配"比"项目很顺利"有价值一百倍。

---

## 卡住怎么办

| 现象 | 原因 | 解法 |
|---|---|---|
| `uvicorn: command not found` | 虚拟环境没激活 | 命令行前面要有 `(.venv)` |
| 端口被占用 | 8000 有别的程序 | 换 `--port 8001`，同时改 `vite.config.js` 里的 target |
| 浏览器控制台报 CORS | 代理没配 或 后端没加中间件 | 两处都要有 |
| 改了 `.env` 没生效 | 配置是启动时读的 | 重启 uvicorn |
| `npm install` 卡住 | 默认源慢 | `npm config set registry https://registry.npmmirror.com` |
| `/docs` 打不开 | 服务没起来 | 看终端有没有报错，别只看浏览器 |

---

## 今天结束你应该达到的状态

**整个项目大约完成 10%。**

- ✅ 能指着每一行代码说清楚它为什么存在
- ✅ 能回答：为什么 POST、为什么 async、Pydantic 干嘛的、跨域怎么来的
- ✅ 有一个自己从零敲出来的仓库

**明天（Day 2）**：加多轮对话历史 + 流式输出。
**暂时不要碰** RAG、LangGraph、MCP——现在碰只会让你更懵。
