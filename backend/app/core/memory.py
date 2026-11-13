"""会话记忆（Day 2）。

核心认知：**模型本身是无状态的。** 它不记得你上一句说了什么，
所以"多轮对话"的本质是——每次请求都把完整历史重新发一遍。

因此"记忆"不在模型那边，在我们这边。存哪儿、存多少、怎么裁，
全都是工程问题，也正是面试官会追问的地方。

本模块只做进程内存版；Day 13 会在这里加历史摘要与长期记忆。

面试要点：
- 为什么用 `deque(maxlen=N)`？淘汰是 O(1)，不用自己写 if len>N: pop(0)（那是 O(n)）。
- 为什么 maxlen 是 max_turns * 2？一轮 = user + assistant 两条消息。
- 多会话隔离怎么保证？按 session_id 分桶，各桶独立 deque，天然不互相污染。
"""

from __future__ import annotations

import uuid
from collections import deque
from typing import Iterable

Message = dict[str, str]


class ConversationMemory:
    """进程内会话存储。

    注意：这是**单进程内存**实现，重启即清空、多 worker 不共享。
    生产环境要换 Redis / 数据库 —— 面试被问到"你的记忆存哪儿"，
    主动说出这个局限比被追问出来要好。
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        # 一轮对话 = 1 条 user + 1 条 assistant，所以容量是轮数 × 2
        self._capacity = max_turns * 2
        self._store: dict[str, deque[Message]] = {}

    # ---------- 会话生命周期 ----------

    @staticmethod
    def new_session_id() -> str:
        return f"sess_{uuid.uuid4().hex[:12]}"

    def ensure(self, session_id: str | None) -> str:
        """前端没传 session_id 就新建一个，传了就复用（并确保桶存在）。"""
        if not session_id:
            session_id = self.new_session_id()
        self._store.setdefault(session_id, deque(maxlen=self._capacity))
        return session_id

    def clear(self, session_id: str) -> bool:
        return self._store.pop(session_id, None) is not None

    def sessions(self) -> list[str]:
        return list(self._store.keys())

    # ---------- 读写 ----------

    def history(self, session_id: str) -> list[Message]:
        """取出该会话的历史（返回副本，避免外部改坏内部状态）。"""
        return list(self._store.get(session_id, ()))

    def append(self, session_id: str, role: str, content: str) -> None:
        bucket = self._store.setdefault(session_id, deque(maxlen=self._capacity))
        bucket.append({"role": role, "content": content})

    def extend(self, session_id: str, messages: Iterable[Message]) -> None:
        for m in messages:
            self.append(session_id, m["role"], m["content"])

    def turn_count(self, session_id: str) -> int:
        """已保留的对话轮数（向下取整，因为一轮含两条消息）。"""
        return len(self._store.get(session_id, ())) // 2

    # ---------- 上下文构造 ----------

    def build_context(
        self,
        session_id: str,
        user_message: str,
        system_prompt: str | None = None,
    ) -> list[Message]:
        """拼出本次要发给模型的完整 messages。

        顺序很关键：system 永远在最前，然后是历史，最后是本轮提问。
        system 放在最后会被模型忽略——这是新手常见错误。
        """
        messages: list[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.history(session_id))
        messages.append({"role": "user", "content": user_message})
        return messages
