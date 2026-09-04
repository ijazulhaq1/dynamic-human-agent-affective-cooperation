"""G_t and OutcomeBaseline — §8, §20.4 of the frozen specification, §5.3 of the Implementation Blueprint."""

from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from models.enums import GoalUpdateSource


def _validate_priority_range(priorities: Mapping[str, Any]) -> dict[str, float]:
    """Shared [0,1] range check for value_priorities values (documented as
    dict[str, float 0..1] in the blueprint's §5.3 table, but not previously
    enforced — implementation-review fix). Used by both GoalState and
    OutcomeBaseline so the two can never quietly diverge on this rule."""
    validated: dict[str, float] = {}
    for key, raw in priorities.items():
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"value_priorities[{key!r}] = {value} is outside the required [0,1] range")
        validated[str(key)] = value
    return validated


class GoalState(BaseModel):
    """G_t. Mutable only via with_explicit_update — see below."""

    objective: str
    value_priorities: dict[str, float] = Field(default_factory=dict)   # feeds D_t's d_goal formula
    stakes: float = Field(ge=0, le=1)                                   # task consequentiality
    task_constraints: dict[str, Any] | list[Any] = Field(default_factory=dict)  # rules, deadlines, evidence reqs
    autonomy_weight: float = Field(ge=0, le=1)      # >= 0.80 triggers AUTONOMY_HIGH (§20.8)
    safety_risk: float = Field(ge=0, le=1)          # >= 0.70 triggers R_SAFETY/TF_SAFETY
    no_undue_influence: bool = True                 # hard constraint, always attached (§20.7 preamble)
    goal_version: int = Field(default=1, ge=1)       # incremented on every allowed update
    update_source: GoalUpdateSource = GoalUpdateSource.RESEARCHER_CONFIG   # logged provenance

    @field_validator("value_priorities")
    @classmethod
    def _value_priorities_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_priority_range(value)

    def with_explicit_update(self, source: GoalUpdateSource, **changes: Any) -> "GoalState":
        """The ONLY sanctioned mutation path (§8.1, §20.4). No affective cue
        alone may call this; callers are the human-explicit-input handler and
        the task-event handler only (services.goal_state_manager, a later
        phase). Always increments goal_version and records update_source, and
        never mutates self — returns a new GoalState.

        Goes through model_validate() rather than model_copy(update=...)
        (implementation-review fix): model_copy's update= does NOT revalidate
        — self.model_copy(update={"safety_risk": 2.0}) would silently accept
        an out-of-range value and produce an invalid GoalState. Reconstructing
        through validation means every changed field is checked exactly as if
        it were a fresh construction.
        """
        data = self.model_dump()
        data.update(changes)
        data["goal_version"] = self.goal_version + 1
        data["update_source"] = source
        return GoalState.model_validate(data)


class OutcomeBaseline(BaseModel):
    """Pre-interaction value-priority profile (§20.4, §20.19). Elicited before
    turn 1, frozen for the run's lifetime. Used ONLY for the primary
    decision-quality outcome measure — never read by AppraisalEstimator,
    DerivedFeatureService, StateTransitionEngine or PolicyEngine, which all
    read GoalState.value_priorities instead."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    # frozen=True (Pydantic v2) blocks *reassigning* a field, e.g.
    # baseline.value_priorities = {...} — but a plain dict is still mutable
    # in place: baseline.value_priorities["safety"] = 0.0 would silently
    # succeed with frozen=True alone (implementation-review fix). Storing a
    # MappingProxyType instead (below) makes THAT mutation raise too, so
    # neither replacing the field nor mutating an entry inside it can succeed.

    run_id: str
    value_priorities: Mapping[str, float]
    elicited_before_interaction: bool = True

    @field_validator("value_priorities")
    @classmethod
    def _freeze_and_validate_priorities(cls, value: Mapping[str, Any]) -> MappingProxyType:
        return MappingProxyType(_validate_priority_range(value))

    @field_serializer("value_priorities")
    def _serialize_priorities(self, value_priorities: Mapping[str, float]) -> dict[str, float]:
        # MappingProxyType isn't natively JSON-serializable; TurnLogger and
        # any other JSON consumer should see a plain dict.
        return dict(value_priorities)
