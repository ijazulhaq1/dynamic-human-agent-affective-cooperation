"""ui/trajectory_view.py — §8's UI component map: "TurnRecord history for
the run — Turn-by-turn trend table/mini-chart." Blueprint §10 Phase 7.

Judgment call (documented gap, flagged for review): same caveat as every
other ui/*.py module in this phase — the blueprint names the component and
its one data source (§8) but no layout. Reads SessionState.
history_for_condition() (services/state_manager.py, this same phase) for
each condition being compared, rather than re-deriving turn history from
anywhere else — one source of truth for "this session's own turn-by-turn
record", matching SessionState's own stated purpose.
"""

from __future__ import annotations

import streamlit as st

from models.enums import Condition
from services.state_manager import SessionState


def render(session: SessionState, conditions: list[Condition]) -> None:
    st.subheader("Trajectory")
    if not session.comparisons:
        st.caption("No turns recorded yet.")
        return

    tabs = st.tabs([c.value for c in conditions]) if conditions else []
    for tab, condition in zip(tabs, conditions):
        history = session.history_for_condition(condition)
        with tab:
            if not history:
                st.caption("(not run in any comparison yet)")
                continue
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
            st.dataframe(rows, width="stretch")
            if condition is not Condition.TASK_FOCUSED and len(history) > 1:
                chart_data = {
                    "turn_id": rows["turn_id"],
                    "a_intv": rows["a_intv"],
                }
                st.line_chart(chart_data, x="turn_id", y="a_intv")
