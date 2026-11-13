"""接口测试（Day 1 起步，Day 2/4/5/10/11/20 逐步补齐）。

面试里「你怎么保证质量」直接拿这个说：
不是"我写了测试"，而是"我知道该测哪些边界"。

本文件的测试全部**离线可跑**（mock LLM + hash 向量），不依赖任何 API key。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # 用 with 才会触发 lifespan（启动时自动建索引）
    with TestClient(app) as c:
        yield c


# ============================================================================
# Day 1：基础闭环
# ============================================================================

def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "app" in resp.json()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]
    # Day 19 新增：健康检查要能告诉调用方「我能不能用」
    assert "llm_provider" in body
    assert "rag_indexed" in body


def test_chat_returns_reply(client):
    resp = client.post("/api/chat", json={"message": "你好，介绍一下你自己"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"]
    assert body["latency_ms"] >= 0
    assert body["model"]
    assert body["session_id"]


def test_chat_rejects_empty_message(client):
    resp = client.post("/api/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_rejects_missing_message(client):
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 422


def test_chat_rejects_unknown_mode(client):
    resp = client.post("/api/chat", json={"message": "hi", "mode": "不存在的模式"})
    assert resp.status_code == 422


# ============================================================================
# Day 2：多轮记忆与隔离
# ============================================================================

def test_chat_auto_generates_session_id(client):
    resp = client.post("/api/chat", json={"message": "第一句"})
    assert resp.status_code == 200
    assert resp.json()["session_id"].startswith("sess_")


def test_chat_remembers_history(client):
    """同一 session 连发两条，第二条要能看到历史。"""
    first = client.post("/api/chat", json={"message": "我叫谷增涛"}).json()
    sid = first["session_id"]

    second = client.post("/api/chat", json={"message": "我叫什么", "session_id": sid}).json()

    # mock 模式会把上下文条数写进回复，用它验证历史确实被带上了
    assert "条消息" in second["reply"]
    assert "约 1 轮历史" in second["reply"]


def test_sessions_are_isolated(client):
    """⭐ 最重要的一条：两个 session 的历史绝不能互相污染。

    这条测试的价值在于：它验证的是**隔离性**，而不是某个功能能不能用。
    多租户系统里数据串了是灾难级 bug，所以必须显式测。
    """
    a = client.post("/api/chat", json={"message": "A 的第一句"}).json()["session_id"]
    b = client.post("/api/chat", json={"message": "B 的第一句"}).json()["session_id"]

    assert a != b

    for _ in range(3):
        client.post("/api/chat", json={"message": "A 继续说", "session_id": a})

    info_a = client.get(f"/api/sessions/{a}").json()
    info_b = client.get(f"/api/sessions/{b}").json()

    assert info_a["turns"] == 4  # 1 + 3
    assert info_b["turns"] == 1  # 不受 A 影响


def test_session_not_found(client):
    assert client.get("/api/sessions/sess_不存在").status_code == 404


def test_clear_session(client):
    sid = client.post("/api/chat", json={"message": "临时会话"}).json()["session_id"]
    assert client.delete(f"/api/sessions/{sid}").json()["cleared"] is True
    assert client.get(f"/api/sessions/{sid}").status_code == 404


# ============================================================================
# Day 8-11：RAG
# ============================================================================

def test_rag_index(client):
    resp = client.post("/api/rag/index", json={"rebuild": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_chunks"] > 0
    assert body["strategy"] in {"fixed", "recursive", "semantic"}


def test_rag_stats(client):
    body = client.get("/api/rag/stats").json()
    assert body["indexed"] is True
    assert body["chunks"] > 0


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_rag_search_all_modes(client, mode):
    resp = client.post(
        "/api/rag/search",
        json={"query": "RAG 是什么", "top_k": 3, "mode": mode},
    )
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits, f"{mode} 检索没有返回结果"
    assert hits[0]["text"]
    assert hits[0]["source"]


def test_rag_search_rejects_bad_mode(client):
    resp = client.post("/api/rag/search", json={"query": "x", "mode": "乱写的"})
    assert resp.status_code == 422


def test_rag_chat_returns_sources(client):
    resp = client.post("/api/chat", json={"message": "什么是 RAG", "mode": "rag"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "rag"
    assert body["reply"]
    # 有出处 = 可核查 = 面试加分项
    assert len(body["sources"]) > 0
    assert body["sources"][0]["source"]


# ============================================================================
# Day 4-5：工具与 Agent
# ============================================================================

def test_tools_listed(client):
    body = client.get("/api/tools").json()
    names = {t["name"] for t in body["tools"]}
    assert {"get_current_time", "calculator", "search_docs", "read_file"} <= names


def test_agent_calls_calculator(client):
    resp = client.post("/api/chat", json={"message": "123 * 456 等于多少", "mode": "agent"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "agent"
    assert body["stopped_by"] == "answer"
    tools_used = [t["tool"] for t in body["trace"]]
    assert "calculator" in tools_used
    assert "56088" in body["reply"]


def test_agent_answers_without_tool(client):
    """闲聊不该调工具 —— 这条很多人做不过。"""
    resp = client.post("/api/chat", json={"message": "你好呀", "mode": "agent"})
    body = resp.json()
    assert body["stopped_by"] == "answer"
    assert body["trace"] == []


def test_tool_manual_call(client):
    resp = client.post("/api/tools/calculator/call", json={"arguments": {"expression": "2+3*4"}})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "14" in resp.json()["result"]


def test_tool_rejects_unknown_name(client):
    resp = client.post("/api/tools/危险函数/call", json={"arguments": {}})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert "未注册" in resp.json()["error"]


def test_tool_rejects_bad_args(client):
    """参数校验失败也要优雅返回，不能 500。"""
    resp = client.post("/api/tools/calculator/call", json={"arguments": {"expression": ""}})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


# ============================================================================
# Day 20：安全 —— 路径穿越必须被拦
# ============================================================================

@pytest.mark.parametrize(
    "evil_path",
    [
        "../../../etc/passwd",
        "..\\..\\Windows\\win.ini",
        "../../.env",
        "....//....//etc/passwd",
    ],
)
def test_read_file_blocks_traversal(client, evil_path):
    """⭐ 路径穿越拦截。面试演示 Prompt Injection 防御时直接跑这个。"""
    resp = client.post("/api/tools/read_file/call", json={"arguments": {"path": evil_path}})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert "越界" in resp.json()["error"] or "不存在" in resp.json()["error"]


def test_calculator_blocks_code_injection(client):
    """用 ast 白名单而不是 eval() —— 这条测试证明它真的挡住了。"""
    resp = client.post(
        "/api/tools/calculator/call",
        json={"arguments": {"expression": "__import__('os').system('echo pwned')"}},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


# ============================================================================
# Day 18：流式
# ============================================================================

def test_stream_chat(client):
    with client.stream("POST", "/api/chat/stream", json={"message": "你好"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())

    assert '"type": "meta"' in raw
    assert '"type": "token"' in raw
    assert '"type": "done"' in raw


def test_stream_rag_sends_sources(client):
    with client.stream(
        "POST", "/api/chat/stream", json={"message": "什么是 RAG", "mode": "rag"}
    ) as resp:
        raw = "".join(resp.iter_text())
    assert '"type": "sources"' in raw
    assert '"type": "done"' in raw
