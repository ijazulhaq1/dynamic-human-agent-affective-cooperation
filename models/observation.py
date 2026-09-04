"""O_t and Turn — §6.2/§20.2 of the frozen specification, §5.1 of the Implementation Blueprint."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """Lightweight, NON-recursive history item.

    Appendix A / §20.2 already names this type: 'raw_history: list[Turn]'.
    This class realizes that name exactly, rather than letting
    Observation.raw_history hold list[Observation] — which would nest a full
    Observation, task_context and all, inside every later Observation's own
    history (unbounded and recursive; the bug the earlier blueprint draft had
    before implementation review).
    """

    turn_id: int = Field(ge=1)
    user_text: str
    task_event: dict[str, Any] | None = None
    explicit_choice: str | None = None
    explicit_preference_change: dict[str, Any] | None = None

    @classmethod
    def from_observation(cls, o_t: "Observation") -> "Turn":
        """Used by Pipeline._commit_shared_observation_once (services layer, later
        phase) to turn a committed O_t into its lightweight history record."""
        return cls(
            turn_id=o_t.turn_id,
            user_text=o_t.user_text,
            task_event=_extract_task_event(o_t.task_context),
            explicit_choice=o_t.observed_choice,
            explicit_preference_change=_extract_pref_change(o_t.task_context),
        )


def _extract_task_event(task_context: dict[str, Any]) -> dict[str, Any] | None:
    """Pulls the objective, non-affective task event (if any) out of
    task_context for storage in a Turn. None when the turn carried no event."""
    event = task_context.get("task_event")
    return dict(event) if event is not None else None


def _extract_pref_change(task_context: dict[str, Any]) -> dict[str, Any] | None:
    """Pulls an explicit, stated preference change (if any) out of task_context."""
    change = task_context.get("explicit_preference_change")
    return dict(change) if change is not None else None


class Observation(BaseModel):
    """O_t — §6, §20.2.

    Built ONLY by services.observation_builder.ObservationBuilder (M1, a later
    phase) so turn_id sequencing, timestamping and history-snapshotting all
    happen in exactly one auditable place. raw_history is the caller's
    pre-turn Turn snapshot — populated by the builder, never left empty.
    """

    turn_id: int = Field(ge=1)
    timestamp: datetime
    user_text: str
    task_context: dict[str, Any] = Field(default_factory=dict)
    observed_choice: str | None = None
    observed_confidence: float | None = Field(default=None, ge=0, le=1)
    raw_history: list[Turn] = Field(default_factory=list)
    event_flags: list[str] = Field(default_factory=list)
