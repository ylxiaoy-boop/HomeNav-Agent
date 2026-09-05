"""Shared data types for long-term household memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    memory_type: str
    content: dict[str, Any]
    confidence: float
    verify_count: int
    created_at: str
    updated_at: str
    access_count: int
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "content": self.content,
            "confidence": self.confidence,
            "verify_count": self.verify_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "score": self.score,
        }
