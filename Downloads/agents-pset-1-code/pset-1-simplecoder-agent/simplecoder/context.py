# Author: Blessing Ndeh
# Date: 03/02/2026
#  Class: AI Agents: COSC 89.34

"""Context management for conversation history"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Message:
    """
    Represents a single message in the conversation.

    Role types:
    - "system": Instructions to the LLM
    - "user": Human input
    - "assistant": LLM's reasoning and actions
    - "tool": Tool execution results
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Convert to OpenAI chat format"""
        return {
            "role": self.role,
            "content": self.content
        }

    def estimate_tokens(self) -> int:
        """
        Estimate token count for this message.

        Rule of thumb: 1 token ≈ 4 characters (or ~0.75 words)
        This is approximate but good enough for management.
        """
        # Simple estimation: characters / 4
        char_count = len(self.content)
        return char_count // 4


class ContextManager:
    """
    Manages conversation history with automatic compacting.

    Responsibilities:
    1. Store all messages (system, user, assistant, tool)
    2. Estimate token usage (avoid API limits)
    3. Compact context when approaching limit (keep important parts)
    4. Preserve recent messages and critical context
    """

    def __init__(
        self,
        max_tokens: int = 6000,  # Leave buffer below actual limit
        compact_threshold: float = 0.8  # Compact when 80% full
    ):
        """
        Initialize context manager.

        Args:
            max_tokens: Maximum tokens to maintain (should be below model limit)
            compact_threshold: Fraction of max_tokens that triggers compacting
        """
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to the context.

        Args:
            role: Message role (system/user/assistant/tool)
            content: Message content
            metadata: Optional metadata dict
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)

        # Check if we need to compact
        if self._should_compact():
            self._compact()

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in OpenAI chat format.

        Returns:
            List of message dicts: [{"role": "user", "content": "..."}, ...]
        """
        return [msg.to_dict() for msg in self.messages]

    def estimate_total_tokens(self) -> int:
        """
        Estimate total token count of all messages.

        Returns:
            Approximate token count
        """
        return sum(msg.estimate_tokens() for msg in self.messages)

    def _should_compact(self) -> bool:
        """
        Check if we should compact the context.

        Returns:
            True if current tokens exceed threshold
        """
        current_tokens = self.estimate_total_tokens()
        threshold_tokens = self.max_tokens * self.compact_threshold
        return current_tokens > threshold_tokens

    def _compact(self):
        """
        Compact the context by removing middle messages.

        Strategy:
        1. Always keep first message (system prompt)
        2. Always keep last N messages (recent context)
        3. Remove middle messages until under threshold
        """
        if len(self.messages) <= 3:
            # Too few messages to compact
            return

        # Keep first message (system prompt)
        first_message = self.messages[0]

        # Keep last 5 messages (recent context)
        keep_last = 5
        recent_messages = self.messages[-keep_last:]

        # Calculate tokens we need to remove
        current_tokens = self.estimate_total_tokens()
        target_tokens = int(self.max_tokens * 0.6)  # Compact to 60%
        tokens_to_remove = current_tokens - target_tokens

        if tokens_to_remove <= 0:
            return

        # Remove middle messages
        middle_messages = self.messages[1:-keep_last]

        # Remove from middle until we're under target
        removed_tokens = 0
        new_middle = []

        for msg in middle_messages:
            if removed_tokens < tokens_to_remove:
                removed_tokens += msg.estimate_tokens()
            else:
                new_middle.append(msg)

        # Reconstruct message list
        self.messages = [first_message] + new_middle + recent_messages

        # Add a summary message indicating compaction happened
        summary = Message(
            role="system",
            content=f"[Context compacted: Removed {removed_tokens} tokens of older messages to stay within limits]"
        )
        # Insert after system prompt
        self.messages.insert(1, summary)

    def clear(self):
        """Clear all messages"""
        self.messages.clear()

    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """
        Get the last N messages.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of last N messages in chat format
        """
        return [msg.to_dict() for msg in self.messages[-n:]]