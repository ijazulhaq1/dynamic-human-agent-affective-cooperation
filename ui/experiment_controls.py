"""ui/experiment_controls.py — condition selection, the temporal-persistence
(rho) controls, counterfactual replay, and session reset.

Every function here is a pure function of one SessionState: it reads only
enough of it to populate widget defaults and never calls any service
itself. Every control returns a plain value/None, and app.py — the one
caller — is the only place that actually calls SessionState/
ExperimentController methods.

Offline and live (OpenAI/Anthropic) backends need different controls, so
app.py calls these independently rather than through one shared panel:

  - render_condition_selector() / render_temporal_persistence() — live
    backends only. Offline's own "Run Prepared Demo" always compares every
    condition, so neither control applies there.
  - render_counterfactual_replay(session) — used by both backends, but
    only once at least one DYNAMIC-condition turn has been recorded.
    DYNAMIC-only: rho only has a causal role for DYNAMIC in the first
    place, so there is no condition selector here. Pipeline.replay_turn()
    itself stays fully condition-general; this function just never calls
    it with anything but Condition.DYNAMIC.
  - render_reset_button() — always available, independent of backend or
    whether any turn has been recorded yet.

rho is the one variable this codebase can vary in isolation without
changing user text or H_t, via Pipeline.replay_turn() (services/
pipeline.py) — it reuses the stored H_t/c_t/D_t/A*_t for a turn and only
re-runs persistence-apply → policy-select → generate at a different rho,
never touching G_t and never mutating the original TurnRecord.
"""

from __future__ import annotations

import streamlit as st

from models.enums import Condition
from services.state_manager import SessionState
from ui.condition_labels import CONDITION_LABELS

ALL_CONDITIONS = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]


def render_condition_selector() -> list[Condition]:
    """Live backends only — which conditions the next live turn's
    compare() call runs, and which conditions the result views display for
    as long as this selection stands. Returns the selection as-is,
    including empty (the caller is responsible for validating that and
    disabling submission rather than this function silently substituting
    every condition)."""
    return st.multiselect(
        "Conditions to compare", options=ALL_CONDITIONS, default=ALL_CONDITIONS,
        format_func=lambda c: CONDITION_LABELS[c],
    )


def render_temporal_persistence() -> float | None:
    """Live backends only — an opt-in override of DYNAMIC's rho for the
    very next live turn. A normal turn uses the configured rho_dynamic by
    default; overriding it is deliberate, unlike render_counterfactual_
    replay()'s always-visible slider below, where adjusting rho is the
    entire point. Returns None when not overridden; app.py passes this
    straight through to ExperimentController.compare()'s rho_override."""
    st.markdown("**Temporal Persistence**")
    st.caption("Adjust ρ to control how much of the previous agent state carries into the current turn.")
    if st.checkbox("Override Dynamic ρ for next turn"):
        return st.slider("Persistence ρ", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    return None


def render_counterfactual_replay(session: SessionState) -> tuple[int, Condition, float] | None:
    """DYNAMIC-only counterfactual rho tool. Renders nothing at all — not
    even a placeholder — until session.history_for_condition(DYNAMIC) has
    at least one record; the caller can therefore invoke this
    unconditionally once any turn exists.

    The turn selector is filtered to turns that actually have a DYNAMIC
    record (a live turn can have DYNAMIC deselected). Once a turn is
    chosen, its original rho is shown read-only (a fact about a past turn,
    not something this tool changes) beside a "Counterfactual ρ" slider
    that defaults to 0.0 — asking "what happens if persistence is
    removed?" is the more useful starting question for a demonstration
    than starting at "no change." The slider carries no explicit `key=`,
    so it does not retain a stale value across a different turn selection."""
    dynamic_history = session.history_for_condition(Condition.DYNAMIC)
    if not dynamic_history:
        return None
    dynamic_records_by_turn = {record.turn_id: record for record in dynamic_history}
    turn_ids = [record.turn_id for record in dynamic_history]

    def _turn_option_label(turn_id: int) -> str:
        text = dynamic_records_by_turn[turn_id].observation.user_text.strip()
        preview = text if len(text) <= 40 else text[:39] + "…"
        return f"Turn {turn_id} — {preview}"

    st.subheader("Counterfactual Analysis")
    st.caption("Re-run a recorded Dynamic turn with a different ρ while preserving the original result.")
    replay_turn_id = st.selectbox("Turn to replay", options=turn_ids, format_func=_turn_option_label)
    original_record = dynamic_records_by_turn[replay_turn_id]

    # Shows which recorded turn is about to be analyzed as soon as the
    # selection changes, before "Run Counterfactual" is ever clicked — the
    # rest of the screen (Latest Interaction, Researcher dashboard) always
    # stays on the most recent actual turn regardless of this selection, so
    # this is the only place that surfaces the selected turn's own text.
    st.markdown("**Selected recorded turn**")
    st.markdown(f"**Participant:** {original_record.observation.user_text}")

    col_original, col_counterfactual = st.columns(2)
    with col_original:
        st.metric("Original ρ", f"{original_record.rho:.3f}")
    with col_counterfactual:
        counterfactual_rho = st.slider("Counterfactual ρ", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    if st.button("Run Counterfactual", key="run_counterfactual"):
        return (replay_turn_id, Condition.DYNAMIC, counterfactual_rho)
    return None


def render_reset_button() -> bool:
    """Always available — a researcher may want to abandon and rebuild the
    current session before any turn has been run, not only after."""
    return st.button("Reset Session", type="primary", key="reset_session")
