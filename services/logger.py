"""TurnLogger — services/logger.py. Blueprint §6.8: "append-only JSONL
writer keyed by record_id ... Mirrors the TurnRecord schema field-for-
field; an optional SQLite mirror ... supports later query-based analysis."

Judgment call (documented gap, flagged for review): no class body or
method signature is given beyond that one bullet and the turn-execution-
flow table's single reference to "TurnLogger.persist" (§7 step 9). The
SQLite mirror is explicitly "optional" and given no schema anywhere in the
document — deferred entirely here, same treatment this project already
gave demo_fixture.yaml's content (Phase 3/4 README notes): not fabricated
ahead of a real spec. persist() below does two things, both defensible
reads of the one sentence given:

1. Appends the record as one JSON line to an append-only file, via
   record.model_dump_json() — Pydantic v2's own schema-preserving JSON
   serialization, so "mirrors the TurnRecord schema field-for-field" holds
   by construction (datetime/enum/nested-model encoding all handled by
   Pydantic itself) rather than by hand-written mapping code that could
   drift from the schema over time.
2. Also keeps an in-memory list of every record persisted this process's
   lifetime (self.records). NOT documented in the blueprint — added
   because Phase 5's own test map (test_record_id_unique_and_comparison_
   id_shared, test_compare_commits_shared_observation_once, and others)
   needs a way to inspect exactly what was persisted without re-parsing a
   JSONL file back off disk in every test. This is pure test/audit
   convenience, not a scientific or architectural decision, and does not
   change what actually gets written to disk — the file is still the
   sole durable record.
"""

from __future__ import annotations

from pathlib import Path

from models.turn_record import TurnRecord


class TurnLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[TurnRecord] = []

    def persist(self, record: TurnRecord) -> None:
        """Append-only: never rewrites or truncates the file, never
        mutates a previously-written line. One JSON object per line
        (JSONL), each keyed by its own globally-unique record_id inside
        the object body — record_id is not additionally used as a
        physical file index; the file itself has no separate index (§6.8's
        optional SQLite mirror is where a keyed index would live, and is
        out of scope for this phase — see the module docstring)."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        self.records.append(record)
