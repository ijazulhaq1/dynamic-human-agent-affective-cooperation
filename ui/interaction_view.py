"""ui/interaction_view.py — §8's UI component map: "Observation, R_t —
Task/context, dialogue, participant input, agent response." Blueprint §10
Phase 7.

Judgment call (documented gap, flagged for review): the blueprint names
this component and what it reads/shows (one table row) but gives no layout,
widget choice, or wording — authored here like every other UI/prompt
surface this phase adds (see llm/prompts.py, ui/researcher_dashboard.py's
own docstrings for the same caveat). Read-only: this module never calls
any service, only renders a dict[Condition, TurnRecord] app.py already
produced via ExperimentController.compare().
"""

from __future__ import annotations

import streamlit as st

from models.enums import Condition
from models.turn_record import TurnRecord


def render(latest_comparison: dict[Condition, TurnRecord] | None, conditions: list[Condition]) -> None:
    st.subheader("Interaction")
    if not latest_comparison:
        st.info("No turns yet — run the demo scenario or send a live message to begin.")
        return

    any_record = next(iter(latest_comparison.values()))
    st.markdown(f"**Turn {any_record.turn_id}**")
    st.markdown(f"**Participant:** {any_record.observation.user_text}")
    task_context = any_record.observation.task_context
    if task_context:
        with st.expander("Task/context", expanded=False):
            st.json(task_context)

    st.markdown("**Agent response, by condition:**")
    columns = st.columns(len(conditions)) if conditions else []
    for column, condition in zip(columns, conditions):
        record = latest_comparison.get(condition)
        with column:
            st.markdown(f"*{condition.value}*")
            if record is None:
                st.caption("(not run this turn)")
                continue
            st.write(record.response_text)
