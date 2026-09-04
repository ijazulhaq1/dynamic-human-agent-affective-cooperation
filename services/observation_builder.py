"""M1 — ObservationBuilder. Blueprint §6.1, spec §6/§20.2.

The only place Observation objects are constructed. Owns turn_id sequencing
and timestamping so those two things happen in exactly one auditable place
instead of being duplicated between Pipeline.run_turn() and Pipeline.compare()
(a later phase, §6.7) — both will call ObservationBuilder.build() with their
own pre-turn history_before snapshot rather than building O_t by hand.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from models.observation import Observation, Turn


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _extract_event_flags(task_context: dict[str, Any]) -> list[str]:
    """Pulls objective task events (never affective inference) out of
    task_context for O_t.event_flags (§5.1 model docstring: 'Objective task
    events only'). Mirrors the sibling _extract_task_event/_extract_pref_change
    helpers already used by Turn.from_observation in models/observation.py:
    a missing or malformed key yields an empty list rather than an error,
    since event_flags is optional scenario metadata, not a required field."""
    flags = task_context.get("event_flags", [])
    if not isinstance(flags, list):
        return []
    return [str(flag) for flag in flags]


class ObservationBuilder:
    """M1 (§6, §20.2). The only place Observation objects are constructed."""

    def __init__(self) -> None:
        self._next_turn_id = 1

    def build(
        self,
        user_text: str,
        task_context: dict[str, Any],
        raw_history: tuple[Turn, ...],
        observed_choice: str | None = None,
        observed_confidence: float | None = None,
    ) -> Observation:
        """raw_history is the caller's pre-turn snapshot (Pipeline's
        history_before, §6.7) — this is what actually populates
        O_t.raw_history; the estimator and generator then see the SAME
        prior-turns list, never an empty placeholder (implementation-review
        fix)."""
        o_t = Observation(
            turn_id=self._next_turn_id,
            timestamp=_utcnow(),
            user_text=user_text,
            task_context=task_context,
            observed_choice=observed_choice,
            observed_confidence=observed_confidence,
            event_flags=_extract_event_flags(task_context),
            raw_history=list(raw_history),
        )
        self._next_turn_id += 1
        return o_t

    def reset_turn_counter(self) -> None:
        """Called only by Pipeline.reset() at the start of a new run (§6.7) —
        never mid-run, since turn_id must stay monotonically increasing for
        the lifetime of a single run."""
        self._next_turn_id = 1
