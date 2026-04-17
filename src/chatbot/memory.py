"""
AGRILION — Conversation Memory
================================
Short-term per-session conversation memory with configurable window size.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str        # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Stores the last N messages per session.

    Usage:
        mem = ConversationMemory(max_turns=5)
        mem.add("ses1", "user", "Hello")
        mem.add("ses1", "assistant", "Hi!")
        history = mem.get_history("ses1")   # OpenAI message format
    """

    def __init__(self, max_turns: int = 5, session_ttl_seconds: int = 3600):
        """
        Args:
            max_turns: max user+assistant pairs to keep per session
            session_ttl_seconds: expire sessions after this many seconds of inactivity
        """
        self.max_turns = max_turns
        self.session_ttl = session_ttl_seconds
        self._sessions: Dict[str, List[Message]] = OrderedDict()
        self._last_access: Dict[str, float] = {}

    def add(self, session_id: str, role: str, content: str):
        """Add a message to a session's history."""
        self._cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(Message(role=role, content=content))
        self._last_access[session_id] = time.time()

        # Trim to max_turns (each turn = user + assistant = 2 messages)
        max_msgs = self.max_turns * 2
        if len(self._sessions[session_id]) > max_msgs:
            self._sessions[session_id] = self._sessions[session_id][-max_msgs:]

    def get_history(self, session_id: str) -> List[dict]:
        """Return conversation history in OpenAI message format."""
        self._last_access[session_id] = time.time()
        messages = self._sessions.get(session_id, [])
        return [{"role": m.role, "content": m.content} for m in messages]

    def clear(self, session_id: str):
        """Clear a session's history."""
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._sessions)

    def _cleanup_expired(self):
        """Remove sessions that haven't been accessed within TTL."""
        now = time.time()
        expired = [
            sid for sid, t in self._last_access.items()
            if (now - t) > self.session_ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)
        if expired:
            logger.debug(f"Expired {len(expired)} inactive sessions")
