"""SQLite-backed personal environment memory with confidence maintenance."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.schema import MemoryRecord


class LongTermMemory:
    """Persistent records learned while carrying out household tasks."""

    def __init__(self, db_path: str = "data/memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def remember(
        self,
        memory_type: str,
        content: dict[str, Any],
        confidence: float = 0.6,
    ) -> MemoryRecord:
        if not isinstance(content, dict) or not content:
            raise ValueError("Memory content must be a non-empty object.")
        confidence = max(0.0, min(1.0, float(confidence)))
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO long_term_memory
                    (memory_type, content, confidence, verify_count, created_at, updated_at, access_count)
                VALUES (?, ?, ?, 1, ?, ?, 0)
                """,
                (memory_type, json.dumps(content, ensure_ascii=False), confidence, now, now),
            )
            memory_id = int(cursor.lastrowid)
        return self.get(memory_id)

    def get(self, memory_id: int) -> MemoryRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM long_term_memory WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Memory {memory_id} does not exist.")
        return self._to_record(row)

    def recall(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        if not query.strip():
            return []
        self.decay_stale()
        normalized_limit = max(1, min(int(limit), 50))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM long_term_memory
                WHERE content LIKE ? OR memory_type LIKE ?
                """,
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            records = [self._to_record(row) for row in rows]
            ranked = sorted(records, key=self._score, reverse=True)[:normalized_limit]
            for record in ranked:
                connection.execute(
                    "UPDATE long_term_memory SET access_count = access_count + 1 WHERE id = ?",
                    (record.id,),
                )
        return [self._with_score(record) for record in ranked]

    def update(self, memory_id: int, content: dict[str, Any]) -> MemoryRecord:
        if not isinstance(content, dict) or not content:
            raise ValueError("Updated memory content must be a non-empty object.")
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE long_term_memory SET content = ?, updated_at = ? WHERE id = ?",
                (json.dumps(content, ensure_ascii=False), self._now(), memory_id),
            )
        if result.rowcount == 0:
            raise KeyError(f"Memory {memory_id} does not exist.")
        return self.get(memory_id)

    def verify(self, memory_id: int) -> MemoryRecord:
        record = self.get(memory_id)
        next_count = record.verify_count + 1
        next_confidence = min(0.95, record.confidence + 0.05 * math.sqrt(next_count))
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE long_term_memory
                SET confidence = ?, verify_count = ?, updated_at = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (next_confidence, next_count, self._now(), memory_id),
            )
        return self.get(memory_id)

    def all(self, limit: int = 100) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM long_term_memory ORDER BY updated_at DESC LIMIT ?", (max(1, min(limit, 500)),)
            ).fetchall()
        return [self._with_score(self._to_record(row)) for row in rows]

    def decay_stale(self, days: int = 30, decay: float = 0.02, forget_below: float = 0.1) -> int:
        """Decay old records, then forget records whose confidence is no longer useful."""
        cutoff = datetime.now(timezone.utc).timestamp() - max(days, 1) * 86400
        changed = 0
        with self._connect() as connection:
            rows = connection.execute("SELECT id, confidence, updated_at FROM long_term_memory").fetchall()
            for row in rows:
                updated = self._parse_time(row["updated_at"]).timestamp()
                if updated > cutoff:
                    continue
                new_confidence = max(0.0, float(row["confidence"]) - decay)
                connection.execute(
                    "UPDATE long_term_memory SET confidence = ?, updated_at = ? WHERE id = ?",
                    (new_confidence, self._now(), row["id"]),
                )
                changed += 1
            connection.execute("DELETE FROM long_term_memory WHERE confidence < ?", (forget_below,))
        return changed

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.6,
                    verify_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _parse_time(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _score(self, record: MemoryRecord) -> float:
        age_days = max(0.0, (datetime.now(timezone.utc) - self._parse_time(record.updated_at)).total_seconds() / 86400)
        recency = math.exp(-age_days / 90)
        access = min(1.0, math.log1p(record.access_count) / math.log(11))
        return round(0.7 * record.confidence + 0.2 * recency + 0.1 * access, 4)

    def _with_score(self, record: MemoryRecord) -> MemoryRecord:
        return MemoryRecord(**{**record.to_dict(), "score": self._score(record)})

    @staticmethod
    def _to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            memory_type=str(row["memory_type"]),
            content=json.loads(row["content"]),
            confidence=float(row["confidence"]),
            verify_count=int(row["verify_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            access_count=int(row["access_count"]),
        )
