"""ui/experiment_controls.py — §8's UI component map: "ExperimentController
— Condition selector, rho toggle/override, replay, one-variable
intervention, 'New Run / Reset Demo' button (calls start_new_run(), never
clears session state by hand instead)." Blueprint §10 Phase 7.

Judgment call (documented gap, flagged for review): same as every other
ui/*.py module — the blueprint names this component and its five controls
in one table row (§8) but gives no widget layout or wording. render()
below is a pure function of one SessionState: it reads only enough of it
to populate widget defaults (available conditions, current turn count for
the replay selector) and NEVER calls any service itself — every control it
exposes returns a plain value/None in ExperimentControlsResult, and app.py
(the one caller) is the only place that actually calls SessionState/
ExperimentController methods, matching this project's established
services-vs-UI separation (ui/*.py modules are read-only renderers
everywhere else in this phase too).

Post-delivery fix (user review of the first Phase 7 delivery): the first
draft's "one-variable intervention" control let a researcher change
GoalState.stakes/autonomy_weight/safety_risk mid-run through
GoalStateManager. That demonstrates a DIFFERENT experiment — changing G_t
changes the mechanism upstream of D_t/A*_t/A_t, so a policy change after
such an edit is not isolated to the same causal pathway the frozen
acceptance criterion actually names: "changing a_info, a_intv, or rho can
change P_t without changing user text or H_t." rho is the one variable
this codebase already has a clean, non-mutating way to vary in isolation —
via Pipeline.replay_turn() (§20.14), which reuses the SAME stored
H_t/c_t/D_t/A*_t for a turn and only re-runs persistence-apply → policy-
select → generate at a different rho, never touching G_t, never mutating
the original TurnRecord, and never re-invoking the estimator. That IS the
frozen one-variable (rho) intervention, already fully built and tested in
Phase 5 (services/pipeline.py's own replay_turn(), tests/test_replay.py) —
this fix stops duplicating it under a scientifically different, G_t-
mutating control and instead labels the ALREADY-CORRECT replay-with-rho-
override control below as what it is. SessionState.apply_explicit_goal_
update() (services/state_manager.py) is unchanged and still available to a
programmatic caller for a genuinely different kind of goal-state
intervention (an explicit human preference change, a task event) — it is
simply no longer exposed here under the "one-variable intervention" label,
since that label is reserved for the frozen rho-isolation demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from models.enums import Condition
from services.state_manager import SessionState

ALL_CONDITIONS = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]


@dataclass(frozen=True)
class ExperimentControlsResult:
    conditions: list[Condition]
    rho_override: float | None
    reset_requested: bool
    replay_request: tuple[int, Condition, float | None] | None


def render(session: SessionState) -> ExperimentControlsResult:
    st.subheader("Experiment controls")

    conditions = st.multiselect(
        "Conditions to compare", options=ALL_CONDITIONS, default=ALL_CONDITIONS,
        format_func=lambda c: c.value,
    )

    st.markdown(
        "**One-variable intervention (rho)** — the frozen causal-isolation demonstration: "
        "changing rho alone can change P_t without changing user text or H_t. Applying it to a "
        "brand-new live turn (below) is logged on that turn's own TurnRecord.interventions; "
        "applying it to a PAST turn (further down, \"Replay a past turn\") never mutates that "
        "turn's original TurnRecord — it produces a separate, additively-logged counterfactual "
        "record instead, so the original is always still there to compare against."
    )
    rho_override = None
    if st.checkbox("Override DYNAMIC's rho for the next live turn"):
        rho_override = st.slider("rho override", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    st.markdown("**Replay a past turn (counterfactual rho derivative)**")
    replay_request = None
    turn_ids = sorted({record.turn_id for records in session.comparisons for record in records.values()})
    if turn_ids:
        replay_turn_id = st.selectbox("Turn to replay", options=turn_ids)
        replay_condition = st.selectbox(
            "Condition to replay", options=ALL_CONDITIONS, format_func=lambda c: c.value, key="replay_condition",
        )
        replay_rho: float | None = None
        if replay_condition is Condition.DYNAMIC and st.checkbox("Override rho for this replay"):
            replay_rho = st.slider("Replay rho", min_value=0.0, max_value=1.0, value=0.35, key="replay_rho")
        if st.button("Replay"):
            replay_request = (replay_turn_id, replay_condition, replay_rho)
    else:
        st.caption("No turns recorded yet — nothing to replay.")

    st.divider()
    reset_requested = st.button("New Run / Reset Demo", type="primary")

    return ExperimentControlsResult(
        conditions=conditions or ALL_CONDITIONS,
        rho_override=rho_override,
        reset_requested=reset_requested,
        replay_request=replay_request,
    )
