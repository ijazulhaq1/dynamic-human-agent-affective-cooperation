"""ui/researcher_dashboard.py — shared human appraisal/decision context,
the per-condition agent-mechanism comparison table, collapsed technical
detail, and the counterfactual-replay comparison view.

H_t, c_t, G_t, and D_t are computed once per turn and shared by
construction across every condition compared in the same turn
(Pipeline.compare()) — rendered once here, not once per condition. Agent
state (A*_t/A_t), rho, and policy genuinely differ per condition, so those
render as one table with a column per condition. TASK_FOCUSED never
computes A*_t/A_t (§6.5), so its column shows "—" for every state row.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from models.enums import Condition
from models.turn_record import TurnRecord
from ui.condition_labels import CONDITION_LABELS

_MECHANISM_ROW_LABELS = [
    "Temporal persistence (rho)",
    "Motivational priority (a_mot) — target A*",
    "Information priority (a_info) — target A*",
    "Intervention readiness (a_intv) — target A*",
    "Motivational priority (a_mot) — A_t",
    "Information priority (a_info) — A_t",
    "Intervention readiness (a_intv) — A_t",
    "Primary policy",
    "Secondary policy",
    "state_could_influence_policy",
    "state_did_influence_policy",
]


def _fmt(value: Any, places: int = 2) -> str:
    """Display formatting only — never rounds, mutates, or writes back to
    the value passed in; only the string this module renders is affected."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}"
    return str(value)


def _render_shared_appraisal_context(record: TurnRecord) -> None:
    """H_t, c_t, G_t, D_t for this turn — identical across every condition
    that ran it, so rendered once from whichever condition's record is
    available first."""
    st.markdown("**Shared human appraisal & decision context**")
    st.caption("These appraisal and decision-context values are shared across conditions for the same turn.")

    st.markdown("**Human appraisal (H_t)**")
    if record.appraisal is None:
        st.caption("No appraisal recorded for this turn.")
    else:
        h_t = record.appraisal
        # Every value below is pre-stringified before st.table(): a column
        # mixing float/bool/dict/enum values makes pyarrow (Streamlit's
        # table serializer) fail its first-pass type inference on every
        # rerun; Streamlit recovers, but only after logging a traceback per
        # failed column.
        st.table(
            {
                "field": [
                    "goal_relevance (h_rel)", "goal_congruence", "uncertainty (h_unc)",
                    "perceived_control (h_ctrl)", "agency", "affect_intensity (h_int)",
                    "possible_affect", "evidence_strength", "c_t",
                ],
                "value": [
                    _fmt(h_t.goal_relevance, 2), _fmt(h_t.goal_congruence, 2), _fmt(h_t.uncertainty, 2),
                    _fmt(h_t.perceived_control, 2), _fmt(h_t.agency, 2), _fmt(h_t.affect_intensity, 2),
                    h_t.possible_affect or "—",
                    record.evidence_strength.value if record.evidence_strength else "—",
                    _fmt(record.c_t, 2),
                ],
            }
        )

    st.markdown("**Goal state (G_t)**")
    g_t = record.goal_state
    st.markdown(f"*Objective:* {g_t.objective}")
    st.table(
        {
            "field": ["value_priorities", "stakes", "autonomy_weight", "safety_risk", "goal_version"],
            "value": [
                str(dict(g_t.value_priorities)), _fmt(g_t.stakes, 2), _fmt(g_t.autonomy_weight, 2),
                _fmt(g_t.safety_risk, 2), str(g_t.goal_version),
            ],
        }
    )

    st.markdown("**Derived decision context (D_t)**")
    d_t = record.derived_state
    st.table(
        {
            "field": ["goal_conflict (d_goal)", "evidence_ambiguity (d_amb)", "goal_conflict_source", "ambiguity_source"],
            "value": [
                _fmt(d_t.goal_conflict, 2), _fmt(d_t.evidence_ambiguity, 2),
                d_t.goal_conflict_source, d_t.ambiguity_source,
            ],
        }
    )


def _mechanism_column(record: TurnRecord | None) -> list[str]:
    """One column of the condition-comparison table — rho and every A*/A
    component read straight from the TurnRecord, no derivation.
    TASK_FOCUSED's a_star/a_t are always None (§6.5), so its column is "—"
    for every state row by construction; rho is shown as "—" alongside
    them too, since rho has no operative meaning without a computed state."""
    if record is None:
        return ["(not run this turn)"] * len(_MECHANISM_ROW_LABELS)

    a_star = record.a_star
    a_t = record.a_t
    state_computed = a_star is not None
    p_t = record.policy

    return [
        _fmt(record.rho, 3) if state_computed else "—",
        _fmt(a_star.motivational_priority, 3) if a_star else "—",
        _fmt(a_star.decision_information_priority, 3) if a_star else "—",
        _fmt(a_star.intervention_readiness, 3) if a_star else "—",
        _fmt(a_t.motivational_priority, 3) if a_t else "—",
        _fmt(a_t.decision_information_priority, 3) if a_t else "—",
        _fmt(a_t.intervention_readiness, 3) if a_t else "—",
        p_t.primary.value,
        p_t.secondary.value if p_t.secondary else "—",
        _fmt(p_t.state_could_influence_policy),
        _fmt(p_t.state_did_influence_policy),
    ]


def _render_condition_mechanism_table(
    latest_comparison: dict[Condition, TurnRecord], conditions: list[Condition]
) -> None:
    st.markdown("**Condition-specific agent mechanism**")
    columns: dict[str, list[str]] = {"field": list(_MECHANISM_ROW_LABELS)}
    for condition in conditions:
        columns[CONDITION_LABELS[condition]] = _mechanism_column(latest_comparison.get(condition))
    st.table(columns)


def _fallback_used(record: TurnRecord) -> bool:
    return "FALLBACK" in record.estimator_status.value or "FALLBACK" in record.generator_status.value


def _render_fallback_warning(record: TurnRecord) -> None:
    """Stays visible outside the collapsed Technical Details expander
    exactly when a live-backend call actually fell back; the raw error
    message itself lives inside Technical Details."""
    if not _fallback_used(record):
        return
    provider = (
        "OpenAI" if record.model_id.lower().startswith(("gpt-", "o1", "o3", "o4"))
        else "Anthropic" if record.model_id.lower().startswith("claude")
        else "Live backend"
    )
    st.warning(
        f"{CONDITION_LABELS[record.condition]}: {provider} call failed; deterministic fallback used. "
        "See Technical Details below for the error message."
    )


def _render_technical_details(record: TurnRecord) -> None:
    p_t = record.policy
    st.markdown(f"**{CONDITION_LABELS[record.condition]}**")
    st.table(
        {
            "field": [
                "triggered_rules", "hard_constraints", "rationale_code",
                "run_id", "comparison_id", "record_id", "model_id", "config_hash",
                "estimator_status", "generator_status",
                "estimator_latency_ms", "generator_latency_ms",
            ],
            "value": [
                ", ".join(p_t.triggered_rules) or "—",
                ", ".join(p_t.hard_constraints) or "—",
                p_t.rationale_code.value,
                record.run_id, record.comparison_id or "—", record.record_id,
                record.model_id, record.config_hash,
                record.estimator_status.value, record.generator_status.value,
                _fmt(record.estimator_latency_ms, 2), _fmt(record.generator_latency_ms, 2),
            ],
        }
    )
    if record.error_messages:
        st.caption("Backend diagnostic:")
        for message in record.error_messages:
            st.code(message, language=None)


def _agent_states_differ(a, b) -> bool:
    if (a is None) != (b is None):
        return True
    if a is None and b is None:
        return False
    return (
        (a.motivational_priority, a.decision_information_priority, a.intervention_readiness)
        != (b.motivational_priority, b.decision_information_priority, b.intervention_readiness)
    )


def _replay_explanation(original: TurnRecord, replayed: TurnRecord) -> str:
    """States the specific reason state/policy did or didn't change
    between the original and the counterfactual — a pure comparison of two
    already-computed records, no new Pipeline/service logic."""
    policy_changed = (
        original.policy.primary != replayed.policy.primary
        or original.policy.secondary != replayed.policy.secondary
    )
    if policy_changed:
        return "Changing temporal persistence (rho) changed the resulting policy."
    if _agent_states_differ(original.a_t, replayed.a_t):
        return "Temporal persistence changed the agent state, but not the resulting policy."
    return "No change in state or policy at this rho."


def _render_replay_column(label: str, record: TurnRecord) -> None:
    st.markdown(f"**{label}** (rho = {_fmt(record.rho, 3)})")
    a_star = record.a_star
    a_t = record.a_t
    p_t = record.policy
    st.table(
        {
            "field": [
                "Motivational priority (a_mot) — target A*",
                "Information priority (a_info) — target A*",
                "Intervention readiness (a_intv) — target A*",
                "Motivational priority (a_mot) — A_t",
                "Information priority (a_info) — A_t",
                "Intervention readiness (a_intv) — A_t",
                "Primary policy", "Secondary policy",
            ],
            "value": [
                _fmt(a_star.motivational_priority, 3) if a_star else "—",
                _fmt(a_star.decision_information_priority, 3) if a_star else "—",
                _fmt(a_star.intervention_readiness, 3) if a_star else "—",
                _fmt(a_t.motivational_priority, 3) if a_t else "—",
                _fmt(a_t.decision_information_priority, 3) if a_t else "—",
                _fmt(a_t.intervention_readiness, 3) if a_t else "—",
                p_t.primary.value, p_t.secondary.value if p_t.secondary else "—",
            ],
        }
    )


def render_replay_comparison(original: TurnRecord | None, replayed: TurnRecord) -> None:
    """Renders a counterfactual-replay result: state (A*_t/A_t) and policy
    (P_t) side by side by default, since Pipeline.replay_turn() always
    re-invokes the generator as a side effect but state/policy — not the
    regenerated response text — is the point of this comparison. The
    regenerated text is available inside a collapsed-by-default expander.

    `original` may be None (app.py's _find_original_record() can fail to
    locate the pre-replay record, e.g. if session state was reset between
    recording and replay) — in that case only the counterfactual side
    renders, with no policy/state-change explanation (nothing to compare it
    against)."""
    st.subheader(f"Counterfactual Analysis — turn {replayed.turn_id}, {CONDITION_LABELS[replayed.condition]}")

    if original is None:
        st.caption(
            "Original turn record not found in this session's history — showing the "
            "counterfactual result only."
        )
        _render_replay_column("Counterfactual", replayed)
    else:
        col_orig, col_replay = st.columns(2)
        with col_orig:
            _render_replay_column("Original", original)
        with col_replay:
            _render_replay_column("Counterfactual", replayed)
        st.info(_replay_explanation(original, replayed))

    with st.expander("Regenerated response text (optional)", expanded=False):
        if original is not None:
            st.write("**Original:**", original.response_text)
        st.write("**Counterfactual:**", replayed.response_text)


def render(latest_comparison: dict[Condition, TurnRecord] | None, conditions: list[Condition]) -> None:
    st.subheader("Researcher dashboard")
    if not latest_comparison:
        st.caption("No turn recorded yet.")
        return

    shared_record = next(
        (latest_comparison[c] for c in conditions if latest_comparison.get(c) is not None), None
    )
    if shared_record is not None:
        _render_shared_appraisal_context(shared_record)
        st.divider()

    _render_condition_mechanism_table(latest_comparison, conditions)

    for condition in conditions:
        record = latest_comparison.get(condition)
        if record is not None:
            _render_fallback_warning(record)

    with st.expander("Technical Details", expanded=False):
        for condition in conditions:
            record = latest_comparison.get(condition)
            if record is not None:
                _render_technical_details(record)
