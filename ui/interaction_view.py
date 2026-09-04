"""ui/interaction_view.py — the participant message and each condition's
response, side by side, with its primary/secondary policy shown directly
underneath.

Always the most recently recorded turn (app.py passes session.
latest_comparison()), independent of whatever turn is selected in
Counterfactual Analysis's own "Turn to replay" control — that selector
never changes this section. Read-only: this module never calls any
service, only renders a dict[Condition, TurnRecord] that app.py already
produced via ExperimentController.compare().
"""

from __future__ import annotations

import streamlit as st

from models.enums import Condition
from models.turn_record import TurnRecord
from ui.condition_labels import CONDITION_BLURBS, CONDITION_LABELS


def render(latest_comparison: dict[Condition, TurnRecord] | None, conditions: list[Condition]) -> None:
    st.subheader("Latest Interaction")
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
            st.markdown(f"**{CONDITION_LABELS[condition]}**")
            st.caption(CONDITION_BLURBS[condition])
            if record is None:
                st.caption("(not run this turn)")
                continue
            st.write(record.response_text)
            secondary = record.policy.secondary.value if record.policy.secondary else "—"
            st.markdown(f"**Policy:** {record.policy.primary.value} (secondary: {secondary})")
