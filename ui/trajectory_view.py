"""ui/trajectory_view.py — turn-by-turn state trajectory: a Current-Cue-
vs-Dynamic comparison chart plus one table/chart per condition.

Reads SessionState.history_for_condition() for each condition being
compared, rather than re-deriving turn history from anywhere else.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from models.enums import Condition
from services.state_manager import SessionState
from ui.condition_labels import CONDITION_LABELS


def _fmt(value, places: int = 2) -> str:
    """Display formatting only, matching ui/researcher_dashboard.py's own
    precision convention (2dp appraisal/context, 3dp state variables).
    Duplicated rather than imported — every ui/*.py module here stays
    self-contained. Never touches the numeric `rows` dict st.line_chart()
    reads; only a separately-built, formatted copy goes to st.dataframe()."""
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}"
    return str(value)


def _current_cue_vs_dynamic_series(history: list, field: str, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn_id": [record.turn_id for record in history],
            label: [getattr(record.a_t, field) if record.a_t else None for record in history],
        }
    ).set_index("turn_id")


def _render_current_cue_vs_dynamic_comparison(session: SessionState) -> None:
    """The main cross-condition trajectory comparison for a_info/a_intv.
    TASK_FOCUSED never computes A_t (§6.5), so it has no state to graph and
    is never part of this chart. Renders nothing if either condition has no
    recorded history yet."""
    cc_history = session.history_for_condition(Condition.CURRENT_CUE)
    dyn_history = session.history_for_condition(Condition.DYNAMIC)
    if not cc_history or not dyn_history:
        return

    cc_label = CONDITION_LABELS[Condition.CURRENT_CUE]
    dyn_label = CONDITION_LABELS[Condition.DYNAMIC]
    st.markdown(f"**{cc_label} vs {dyn_label} — a_info and a_intv over time**")
    st.caption(
        f"{cc_label} uses no cross-turn state carryover; {dyn_label} integrates the previous agent state."
    )

    a_info = _current_cue_vs_dynamic_series(
        cc_history, "decision_information_priority", f"a_info ({cc_label})"
    ).join(
        _current_cue_vs_dynamic_series(dyn_history, "decision_information_priority", f"a_info ({dyn_label})"),
        how="outer",
    ).sort_index()
    st.line_chart(a_info)

    a_intv = _current_cue_vs_dynamic_series(
        cc_history, "intervention_readiness", f"a_intv ({cc_label})"
    ).join(
        _current_cue_vs_dynamic_series(dyn_history, "intervention_readiness", f"a_intv ({dyn_label})"),
        how="outer",
    ).sort_index()
    st.line_chart(a_intv)
    st.divider()


def render(session: SessionState, conditions: list[Condition]) -> None:
    st.subheader("State Trajectory")
    if not session.comparisons:
        st.caption("No turns recorded yet.")
        return

    if Condition.CURRENT_CUE in conditions and Condition.DYNAMIC in conditions:
        _render_current_cue_vs_dynamic_comparison(session)

    tabs = st.tabs([CONDITION_LABELS[c] for c in conditions]) if conditions else []
    for tab, condition in zip(tabs, conditions):
        history = session.history_for_condition(condition)
        with tab:
            if not history:
                st.caption("(not run in any comparison yet)")
                continue
            # `rows` stays numeric — st.line_chart() below needs real
            # floats, not display strings.
            rows = {
                "turn_id": [record.turn_id for record in history],
                "policy": [record.policy.primary.value for record in history],
                "c_t": [record.c_t for record in history],
            }
            if condition is not Condition.TASK_FOCUSED:
                rows["a_mot"] = [
                    record.a_t.motivational_priority if record.a_t else None for record in history
                ]
                rows["a_info"] = [
                    record.a_t.decision_information_priority if record.a_t else None for record in history
                ]
                rows["a_intv"] = [
                    record.a_t.intervention_readiness if record.a_t else None for record in history
                ]
            # Display copy only — 2dp for c_t, 3dp for state variables.
            display_rows = {
                "turn_id": rows["turn_id"],
                "policy": rows["policy"],
                "c_t": [_fmt(v, 2) for v in rows["c_t"]],
            }
            for field in ("a_mot", "a_info", "a_intv"):
                if field in rows:
                    display_rows[field] = [_fmt(v, 3) for v in rows[field]]
            st.dataframe(display_rows, width="stretch")
            if condition is not Condition.TASK_FOCUSED and len(history) > 1:
                chart_data = {
                    "turn_id": rows["turn_id"],
                    "a_intv": rows["a_intv"],
                }
                st.line_chart(chart_data, x="turn_id", y="a_intv")
