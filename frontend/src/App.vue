<script setup>
import { ref, nextTick, onMounted, computed } from 'vue'
import { streamMessage, sendMessage, checkHealth, ragStats } from './api'

const MODES = [
  { key: 'chat', label: '普通对话', hint: '多轮记忆，不查资料' },
  { key: 'rag', label: '资料问答', hint: '先检索本地文档，回答带出处' },
  { key: 'agent', label: 'Agent', hint: 'ReAct 循环，可调工具' }
]

const input = ref('')
const messages = ref([])
const loading = ref(false)
const errorMsg = ref('')
const backendOk = ref(null)
const lastLatency = ref(null)
const lastModel = ref('')
const listEl = ref(null)

const mode = ref('chat')
const useStream = ref(true)
const sessionId = ref(null)
const rag = ref(null)

let controller = null

onMounted(async () => {
  try {
    backendOk.value = (await checkHealth()).status === 'ok'
  } catch {
    backendOk.value = false
  }
  try {
    rag.value = await ragStats()
  } catch {
    /* RAG 状态拿不到不影响对话 */
  }
})

const modeHint = computed(() => MODES.find((m) => m.key === mode.value)?.hint ?? '')

async function scrollToBottom() {
  await nextTick()
  if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight
}

function pushAssistant(extra = {}) {
  const msg = { role: 'assistant', content: '', sources: [], trace: [], stoppedBy: null, ...extra }
  messages.value.push(msg)
  return messages.value[messages.value.length - 1]
}

async function handleSend() {
  const text = input.value.trim()
  if (!text || loading.value) return

  errorMsg.value = ''
  messages.value.push({ role: 'user', content: text })
  input.value = ''
  loading.value = true
  await scrollToBottom()

  const target = pushAssistant()
  const started = performance.now()

  try {
    if (useStream.value) {
      await runStream(text, target)
    } else {
      await runBlocking(text, target)
    }
    backendOk.value = true
  } catch (e) {
    errorMsg.value = e.message
    if (!target.content) {
      target.content = '（请求失败，请确认后端已在 8000 端口启动）'
    }
    backendOk.value = false
  } finally {
    loading.value = false
    lastLatency.value = Math.round(performance.now() - started)
    await scrollToBottom()
  }
}

async function runStream(text, target) {
  controller = new AbortController()

  await streamMessage(text, {
    sessionId: sessionId.value,
    mode: mode.value,
    signal: controller.signal,
    onEvent(ev) {
      if (ev.type === 'meta') {
        sessionId.value = ev.session_id
        lastModel.value = ev.model
      } else if (ev.type === 'sources') {
        target.sources = ev.sources ?? []
      } else if (ev.type === 'token') {
        target.content += ev.text
        scrollToBottom()
      } else if (ev.type === 'error') {
        errorMsg.value = ev.message
      }
    }
  })
  controller = null
}

async function runBlocking(text, target) {
  const data = await sendMessage(text, { sessionId: sessionId.value, mode: mode.value })
  sessionId.value = data.session_id
  target.content = data.reply
  target.sources = data.sources ?? []
  target.trace = data.trace ?? []
  target.stoppedBy = data.stopped_by
  lastLatency.value = data.latency_ms
  lastModel.value = data.model
}

function stopStream() {
  controller?.abort()
  controller = null
  loading.value = false
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

function newSession() {
  sessionId.value = null
  messages.value = []
  errorMsg.value = ''
  lastLatency.value = null
}

async function refreshRag() {
  try {
    rag.value = await ragStats()
  } catch {
    rag.value = null
  }
}
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="title">
        <h1>Agent 求职学习助手</h1>
        <p class="subtitle">Vue → FastAPI → RAG → Agent</p>
      </div>
      <div class="header-right">
        <div class="status" :class="backendOk === false ? 'bad' : backendOk ? 'good' : 'unknown'">
          <span class="dot"></span>
          {{ backendOk === null ? '检测中…' : backendOk ? '后端已连接' : '后端未连接' }}
        </div>
        <div class="rag-badge" :class="rag?.indexed ? 'good' : 'bad'" @click="refreshRag">
          {{ rag?.indexed ? `索引 ${rag.chunks} 段` : '索引未建' }}
        </div>
      </div>
    </header>

    <div class="toolbar">
      <div class="modes">
        <button
          v-for="m in MODES"
          :key="m.key"
          class="mode-btn"
          :class="{ active: mode === m.key }"
          @click="mode = m.key"
        >
          {{ m.label }}
        </button>
      </div>
      <div class="toolbar-right">
        <label class="switch">
          <input type="checkbox" v-model="useStream" />
          流式
        </label>
        <button class="ghost" @click="newSession">新会话</button>
      </div>
    </div>
    <p class="mode-hint">{{ modeHint }}</p>

    <main ref="listEl" class="messages">
      <div v-if="messages.length === 0" class="empty">
        <p>先随便问一句，验证链路是否打通。</p>
        <p class="hint">试试切到「资料问答」问：什么是 RAG</p>
        <p class="hint">或切到「Agent」问：现在几点了</p>
      </div>

      <div v-for="(m, i) in messages" :key="i" class="bubble-row" :class="m.role">
        <div class="bubble-wrap">
          <div class="bubble" :class="{ empty: !m.content && loading && m.role === 'assistant' }">
            <template v-if="m.content">{{ m.content }}</template>
            <span v-else-if="loading && m.role === 'assistant'" class="typing">
              <span></span><span></span><span></span>
            </span>
          </div>

          <div v-if="m.sources?.length" class="sources">
            <p class="sources-title">引用来源</p>
            <div v-for="s in m.sources" :key="s.index" class="source-item">
              <span class="source-idx">[{{ s.index }}]</span>
              <span class="source-name">{{ s.source }}<template v-if="s.page"> 第{{ s.page }}页</template></span>
              <span class="source-score">{{ s.score?.toFixed(3) }}</span>
            </div>
          </div>

          <div v-if="m.trace?.length" class="trace">
            <p class="trace-title">
              工具调用轨迹
              <span v-if="m.stoppedBy" class="stop-reason">终止原因：{{ m.stoppedBy }}</span>
            </p>
            <div v-for="(t, ti) in m.trace" :key="ti" class="trace-item" :class="{ fail: !t.ok }">
              <span class="trace-step">#{{ t.step }}</span>
              <span class="trace-tool">{{ t.tool }}</span>
              <span class="trace-args">{{ t.arguments }}</span>
              <span class="trace-time">{{ t.duration_ms }}ms</span>
            </div>
          </div>
        </div>
      </div>
    </main>

    <div v-if="errorMsg" class="error">{{ errorMsg }}</div>

    <footer class="composer">
      <textarea
        v-model="input"
        rows="2"
        placeholder="输入问题，Enter 发送，Shift+Enter 换行"
        @keydown="handleKeydown"
      ></textarea>
      <button v-if="loading && controller" class="stop" @click="stopStream">停止</button>
      <button v-else :disabled="loading || !input.trim()" @click="handleSend">
        {{ loading ? '生成中…' : '发送' }}
      </button>
    </footer>

    <div class="meta" v-if="lastLatency !== null">
      <span v-if="sessionId">会话 <code>{{ sessionId }}</code> · </span>
      <span v-if="lastModel">模型 <code>{{ lastModel }}</code> · </span>
      最近一次 {{ lastLatency }} ms
    </div>
  </div>
</template>

<style scoped>
.app {
  max-width: 860px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 16px;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}

.title h1 { font-size: 18px; margin: 0; }
.subtitle { margin: 2px 0 0; font-size: 12px; color: var(--text-sub); }

.header-right { display: flex; align-items: center; gap: 8px; }

.status, .rag-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--assistant-bubble);
  color: var(--text-sub);
}
.rag-badge { cursor: pointer; }

.status .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--text-sub);
}
.status.good .dot, .rag-badge.good { background: var(--ok); }
.rag-badge.good { color: #fff; }
.status.bad .dot { background: var(--danger); }
.rag-badge.bad { background: var(--danger); color: #fff; }

.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-top: 12px;
  gap: 12px;
}

.modes { display: flex; gap: 6px; }

.mode-btn {
  padding: 6px 14px;
  font-size: 13px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: transparent;
  color: var(--text-sub);
  cursor: pointer;
}
.mode-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

.toolbar-right { display: flex; align-items: center; gap: 10px; }

.switch {
  display: flex; align-items: center; gap: 4px;
  font-size: 12px; color: var(--text-sub); cursor: pointer;
}

.ghost {
  padding: 5px 12px;
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: transparent;
  color: var(--text-sub);
  cursor: pointer;
}

.mode-hint {
  margin: 6px 0 0;
  font-size: 12px;
  color: var(--text-sub);
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px 4px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.empty { margin: auto; text-align: center; color: var(--text-sub); }
.empty .hint { font-size: 12px; }

.bubble-row { display: flex; }
.bubble-row.user { justify-content: flex-end; }
.bubble-row.assistant { justify-content: flex-start; }

.bubble-wrap { max-width: 82%; }

.bubble {
  padding: 10px 14px;
  border-radius: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.bubble-row.user .bubble {
  background: var(--user-bubble);
  color: #fff;
  border-bottom-right-radius: 4px;
}

.bubble-row.assistant .bubble {
  background: var(--assistant-bubble);
  border-bottom-left-radius: 4px;
}

.typing { display: inline-flex; gap: 4px; align-items: center; }
.typing span {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--text-sub);
  animation: blink 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink {
  0%, 80%, 100% { opacity: 0.2; }
  40% { opacity: 1; }
}

.sources, .trace {
  margin-top: 8px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 10px;
  font-size: 12px;
}

.sources-title, .trace-title {
  margin: 0 0 6px;
  color: var(--text-sub);
  font-size: 12px;
  display: flex;
  justify-content: space-between;
}

.stop-reason { color: var(--danger); }

.source-item, .trace-item {
  display: flex;
  gap: 8px;
  align-items: baseline;
  padding: 2px 0;
}

.source-idx { color: var(--accent); font-weight: 600; }
.source-name { flex: 1; word-break: break-all; }
.source-score { color: var(--text-sub); font-variant-numeric: tabular-nums; }

.trace-item.fail { color: var(--danger); }
.trace-step { color: var(--text-sub); }
.trace-tool { font-weight: 600; }
.trace-args { flex: 1; color: var(--text-sub); word-break: break-all; }
.trace-time { font-variant-numeric: tabular-nums; }

.error { color: var(--danger); font-size: 12px; padding: 6px 4px; }

.composer {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}

.composer textarea {
  flex: 1;
  resize: none;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 10px;
  font-family: inherit;
  font-size: 14px;
  outline: none;
}
.composer textarea:focus { border-color: var(--accent); }

.composer button {
  padding: 10px 20px;
  border: none;
  border-radius: 10px;
  background: var(--accent);
  color: #fff;
  font-size: 14px;
  cursor: pointer;
}
.composer button:hover:not(:disabled) { background: var(--accent-hover); }
.composer button:disabled { opacity: 0.5; cursor: not-allowed; }
.composer button.stop { background: var(--danger); }

.meta { font-size: 12px; color: var(--text-sub); padding-top: 8px; }
.meta code {
  background: var(--assistant-bubble);
  padding: 1px 6px;
  border-radius: 4px;
}
</style>
