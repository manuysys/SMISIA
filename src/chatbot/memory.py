"""
AGRILION — Conversation Memory v2
====================================
Improved: deduplication hints, efficient trimming, TTL cleanup.
"""

import time
import logging
from typing import Dict, List, Optional
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Per-session conversation history with:
    - Configurable max turns (default 4 = 8 messages)
    - TTL-based session expiry
    - Deduplication: doesn't store identical consecutive messages
    """

    def __init__(self, max_turns: int = 4, session_ttl_seconds: int = 3600):
        self.max_turns = max_turns
        self.session_ttl = session_ttl_seconds
        self._sessions: Dict[str, List[Message]] = OrderedDict()
        self._last_access: Dict[str, float] = {}

    def add(self, session_id: str, role: str, content: str):
        """Add message; skips if identical to last message in session."""
        self._cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        history = self._sessions[session_id]

        # Dedup: skip if same role+content as last message
        if history and history[-1].role == role and history[-1].content == content:
            return

        history.append(Message(role=role, content=content))
        self._last_access[session_id] = time.time()

        # Trim: keep last max_turns pairs (max_turns * 2 messages)
        limit = self.max_turns * 2
        if len(history) > limit:
            self._sessions[session_id] = history[-limit:]

    def get_history(self, session_id: str) -> List[dict]:
        """Return history in OpenAI message format."""
        self._last_access[session_id] = time.time()
        return [
            {"role": m.role, "content": m.content}
            for m in self._sessions.get(session_id, [])
        ]

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._sessions)

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, t in self._last_access.items()
            if (now - t) > self.session_ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)
        if expired:
            logger.debug(f"Expired {len(expired)} sessions")
