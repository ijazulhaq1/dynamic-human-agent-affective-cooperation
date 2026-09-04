"""M4 — DerivedFeatureService. Blueprint §6.3, spec §8.2/§20.5.

derive_interaction_state implements §20.5 exactly. Its signature has no
h_t parameter at all — D_t is derived from O_t and G_t only, and NEVER
reads H_t (the bug fixed two blueprint review rounds ago: c_t/H_t feeding
D_t). That is checkable structurally here, not merely by convention.
"""

from __future__ import annotations

from typing import Any

from models.derived_state import DerivedInteractionState
from models.goal_state import GoalState
from models.observation import Observation


class ScenarioConfigError(Exception):
    """Raised when a scenario's task_context is internally inconsistent —
    e.g. the focal option impacts a value G_t.value_priorities does not
    weight. This is a scenario-authoring bug, deliberately raised loudly at
    the point of use rather than surfacing as a bare KeyError mid-demo. For
    demo_fixture.yaml (Phase 6) it will additionally be checked at
    fixture-load time via validate_demo_fixture(), before any turn runs."""


def clip01(value: float) -> float:
    """Clip to the closed [0,1] interval. Used everywhere a formula's output
    is only bounded on one side (e.g. d_goal = clip(2 * min(P, N)) can only
    exceed 1, never go negative, since P and N are themselves non-negative
    weighted sums) or not bounded at all without it."""
    return max(0.0, min(1.0, value))


def _validated_impact(value_name: str, impact: Any) -> float:
    """§20.5: each focal-option impact q_j is defined as normalized to
    [-1,1]. Validated here, before the d_goal formula runs, rather than
    relying on the formula's own clip01() to hide an out-of-range or
    non-numeric q_j — clip01() is meant to bound a valid formula's output,
    not silently repair invalid scenario input (implementation-review fix:
    {"a": 2.0, "b": -2.0} previously passed through unchecked and produced
    an indistinguishable-from-legitimate goal_conflict=1.0)."""
    try:
        q = float(impact)
    except (TypeError, ValueError):
        raise ScenarioConfigError(f"impact for {value_name!r} must be numeric, got {impact!r}")
    if not -1.0 <= q <= 1.0:
        raise ScenarioConfigError(f"impact for {value_name!r} must be in [-1,1], got {q}")
    return q


# Canonical required_evidence.status vocabulary (§20.5). A status outside
# this set is rejected rather than silently treated as resolved — the
# implementation-review fix for {"status": "typo"} previously producing
# evidence_ambiguity=0.0 with no error. Standardized on these four lowercase
# labels; do not add a second spelling (e.g. "confirmed") without a reason,
# since d_amb's whole point is that every status is accounted for.
VALID_EVIDENCE_STATUSES = {"resolved", "missing", "contradictory", "unresolved"}
UNRESOLVED_EVIDENCE_STATUSES = {"missing", "contradictory", "unresolved"}


def _normalized_weights(value_priorities: dict[str, float]) -> dict[str, float]:
    """Normalizes G_t.value_priorities so sum_j w_j = 1, per §20.5's d_goal
    formula. Raises ScenarioConfigError rather than dividing by zero when
    every priority is 0 — a goal state that weights nothing cannot support
    a structured d_goal computation, and that is a scenario-authoring
    problem to surface immediately, not a silent NaN/ZeroDivisionError."""
    total = sum(value_priorities.values())
    if total <= 0:
        raise ScenarioConfigError(
            "cannot normalize value_priorities to compute d_goal: "
            f"weights sum to {total} (value_priorities={value_priorities!r})"
        )
    return {key: value / total for key, value in value_priorities.items()}


def derive_interaction_state(o_t: Observation, g_t: GoalState) -> DerivedInteractionState:
    """§20.5 exactly. d_goal from the structured option-impact formula when
    the scenario provides options/focal_option_id, else a task_rule/
    researcher_config override; d_amb from the required-evidence-resolution
    ratio, else the equivalent override."""
    task_context: dict[str, Any] = o_t.task_context
    options: dict[str, dict[str, float]] | None = task_context.get("options")
    focal_id = task_context.get("focal_option_id")

    if options and focal_id is not None:
        if focal_id not in options:
            raise ScenarioConfigError(
                f"focal_option_id {focal_id!r} not among options {list(options)}"
            )
        impacts = options[focal_id]
        unweighted = set(impacts) - set(g_t.value_priorities)
        if unweighted:
            raise ScenarioConfigError(
                f"option {focal_id!r} impacts values {unweighted} that G_t.value_priorities "
                "does not weight — fix the scenario config, not a runtime KeyError mid-demo"
            )
        validated_impacts = {
            value_name: _validated_impact(value_name, impact)
            for value_name, impact in impacts.items()
        }
        w = _normalized_weights(g_t.value_priorities)
        p_support = sum(w[j] * max(validated_impacts[j], 0.0) for j in validated_impacts)
        n_oppose = sum(w[j] * max(-validated_impacts[j], 0.0) for j in validated_impacts)
        d_goal = clip01(2 * min(p_support, n_oppose))
        goal_conflict_source = "structured_formula"
    else:
        d_goal = float(task_context.get("d_goal_override", 0.0))
        goal_conflict_source = "task_rule"

    required = task_context.get("required_evidence", [])
    if required:
        for item in required:
            status = item["status"]
            if status not in VALID_EVIDENCE_STATUSES:
                raise ScenarioConfigError(
                    f"required_evidence status {status!r} is not one of "
                    f"{sorted(VALID_EVIDENCE_STATUSES)} — fix the scenario config, "
                    "not a silently-wrong d_amb"
                )
        unresolved = [item for item in required if item["status"] in UNRESOLVED_EVIDENCE_STATUSES]
        d_amb = len(unresolved) / len(required)
        ambiguity_source = "required_evidence_formula"
    else:
        d_amb = float(task_context.get("d_amb_override", 0.0))
        ambiguity_source = "task_rule"

    return DerivedInteractionState(
        goal_conflict=d_goal,
        evidence_ambiguity=d_amb,
        goal_conflict_source=goal_conflict_source,
        ambiguity_source=ambiguity_source,
    )
