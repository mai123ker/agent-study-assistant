"""会话记忆单元测试（Day 2）。

重点测**滑动窗口**和**隔离性** —— 这两个才是多轮对话真正的坑。
"""

from app.core.memory import ConversationMemory


def test_creates_session_when_missing():
    mem = ConversationMemory()
    sid = mem.ensure(None)
    assert sid.startswith("sess_")
    assert mem.history(sid) == []


def test_reuses_existing_session():
    mem = ConversationMemory()
    sid = mem.ensure("my-session")
    assert sid == "my-session"
    assert mem.ensure("my-session") == "my-session"


def test_append_and_history():
    mem = ConversationMemory()
    sid = mem.ensure(None)
    mem.append(sid, "user", "你好")
    mem.append(sid, "assistant", "你也好")

    assert mem.history(sid) == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你也好"},
    ]
    assert mem.turn_count(sid) == 1


def test_sliding_window_evicts_oldest():
    """⭐ 超过 max_turns 后，最早的对话要被淘汰。

    不淘汰的后果：上下文无限增长 → 变慢、变贵，而且**变傻**
    （长上下文里模型会忽略中间信息，即 lost in the middle）。
    """
    mem = ConversationMemory(max_turns=3)
    sid = mem.ensure(None)

    for i in range(10):
        mem.append(sid, "user", f"问题{i}")
        mem.append(sid, "assistant", f"回答{i}")

    history = mem.history(sid)
    assert len(history) == 6  # 3 轮 × 2 条
    assert history[0]["content"] == "问题7"  # 最早的被淘汰了
    assert history[-1]["content"] == "回答9"
    assert mem.turn_count(sid) == 3


def test_sessions_isolated():
    mem = ConversationMemory()
    a, b = mem.ensure(None), mem.ensure(None)

    mem.append(a, "user", "A 的内容")

    assert len(mem.history(a)) == 1
    assert mem.history(b) == []


def test_history_returns_copy():
    """外部改返回值不该影响内部状态。"""
    mem = ConversationMemory()
    sid = mem.ensure(None)
    mem.append(sid, "user", "原始内容")

    snapshot = mem.history(sid)
    snapshot.append({"role": "user", "content": "注入的"})

    assert len(mem.history(sid)) == 1


def test_build_context_puts_system_first():
    """system 必须在最前 —— 放最后会被模型忽略，是新手常见错误。"""
    mem = ConversationMemory()
    sid = mem.ensure(None)
    mem.append(sid, "user", "历史问题")
    mem.append(sid, "assistant", "历史回答")

    ctx = mem.build_context(sid, "本轮问题", system_prompt="你是助手")

    assert ctx[0]["role"] == "system"
    assert ctx[1]["content"] == "历史问题"
    assert ctx[-1] == {"role": "user", "content": "本轮问题"}
    assert len(ctx) == 4


def test_clear_session():
    mem = ConversationMemory()
    sid = mem.ensure(None)
    mem.append(sid, "user", "x")

    assert mem.clear(sid) is True
    assert mem.clear(sid) is False
    assert mem.history(sid) == []
