// 统一接口层：组件里不直接写 fetch，方便以后加 token、重试、错误上报。
const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options)
  if (!res.ok) {
    let detail = ''
    try {
      detail = JSON.stringify(await res.json())
    } catch {
      detail = await res.text()
    }
    throw new Error(`请求失败 (${res.status}): ${detail}`)
  }
  return res.json()
}

const jsonPost = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body)
})

// ---------------- 对话 ----------------

export function sendMessage(message, { sessionId = null, mode = 'chat' } = {}) {
  return request('/api/chat', jsonPost({ message, session_id: sessionId, mode }))
}

/**
 * 流式对话（SSE）。
 *
 * 为什么不用 EventSource？因为 EventSource 只支持 GET，而我们要 POST 传 body。
 * 所以用 fetch + ReadableStream 手动解析 SSE 帧。
 *
 * @param {object} opts
 * @param {(payload:object)=>void} opts.onEvent 每收到一个事件回调一次
 * @param {AbortSignal} opts.signal 用来中途取消
 */
export async function streamMessage(message, { sessionId = null, mode = 'chat', onEvent, signal } = {}) {
  const res = await fetch(`${BASE_URL}/api/chat/stream`, {
    ...jsonPost({ message, session_id: sessionId, mode }),
    signal
  })

  if (!res.ok) {
    throw new Error(`请求失败 (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // SSE 帧以空行分隔；最后一段可能不完整，留在 buffer 里等下一块
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const raw = line.slice(5).trim()
      if (!raw) continue
      try {
        onEvent?.(JSON.parse(raw))
      } catch {
        // 单帧解析失败不该中断整个流
      }
    }
  }
}

export function checkHealth() {
  return request('/health')
}

// ---------------- RAG ----------------

export function indexDocs(rebuild = true) {
  return request('/api/rag/index', jsonPost({ rebuild }))
}

export function ragStats() {
  return request('/api/rag/stats')
}

export function searchDocs(query, { topK = 3, mode = null } = {}) {
  return request('/api/rag/search', jsonPost({ query, top_k: topK, mode }))
}

// ---------------- 工具 ----------------

export function listTools() {
  return request('/api/tools')
}

export function callTool(name, args) {
  return request(`/api/tools/${encodeURIComponent(name)}/call`, jsonPost({ arguments: args }))
}
