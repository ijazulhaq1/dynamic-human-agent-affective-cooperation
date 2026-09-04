"""ui/researcher_dashboard.py — §8's UI component map, four rows folded
into one module (blueprint §10 Phase 7):

  Human/Goals panel     HumanAppraisal, EvidenceStrength, c_t, GoalState
  Agent State panel     AgentTargetState, AgentState, rho (shown separately,
                         NEVER inside A_t — §20.16)
  Policy panel          PolicyState (primary, secondary, triggered_rules,
                         hard_constraints, rationale_code, both
                         state-relevance flags)
  Status bar            estimator/generator latency, status, config_hash

Judgment call (documented gap, flagged for review): the blueprint's own
component-map table (§8) names exactly these four rows and which fields
each reads — reproduced above verbatim — but gives no layout or wording;
one module with one tab per condition (rather than four separate files) was
chosen here for how directly each panel reads off ONE condition's own
TurnRecord — every field below is read straight from TurnRecord (models/
turn_record.py) with no derivation of its own. Read-only, same as
ui/interaction_view.py.
"""

from __future__ import annotations

import streamlit as st

from models.enums import Condition
from models.turn_record import TurnRecord


def _render_human_goals_panel(record: TurnRecord) -> None:
    st.markdown("**Human appraisal (H_t)**")
    if record.appraisal is None:
        st.caption("No appraisal recorded for this record.")
    else:
        h_t = record.appraisal
        # Judgment call: every "value" column below is str()-cast uniformly
        # before reaching st.table() — a column mixing float/bool/dict/enum
        # values makes pyarrow (Streamlit's own table serializer) fail its
        # first-pass type inference on every rerun; Streamlit auto-recovers
        # by stringifying anyway, but only after logging a full traceback
        # per failed column, which would spam a live researcher's own
        # console during an interview. Pre-stringifying avoids that path
        # entirely while rendering identically.
        st.table(
            {
                "field": [
                    "goal_relevance (h_rel)", "goal_congruence", "uncertainty (h_unc)",
                    "perceived_control (h_ctrl)", "agency", "affect_intensity (h_int)",
                    "possible_affect", "evidence_strength", "c_t",
                ],
                "value": [
                    str(v) for v in (
                        h_t.goal_relevance, h_t.goal_congruence, h_t.uncertainty,
                        h_t.perceived_control, h_t.agency, h_t.affect_intensity,
                        h_t.possible_affect or "—",
                        record.evidence_strength.value if record.evidence_strength else "—",
                        record.c_t,
                    )
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
                str(v) for v in
                (dict(g_t.value_priorities), g_t.stakes, g_t.autonomy_weight, g_t.safety_risk, g_t.goal_version)
            ],
        }
    )


def _render_agent_state_panel(record: TurnRecord) -> None:
    st.markdown("**Agent state**")
    if record.a_star is None:
        st.caption("Not computed for this condition (TASK_FOCUSED never computes A*_t/A_t — §6.5).")
        return
    st.caption(f"rho = {record.rho}  (shown separately — never a field inside A_t itself, §20.2)")
    rows = {
        "field": ["motivational_priority (a_mot)", "decision_information_priority (a_info)",
                  "intervention_readiness (a_intv)"],
        "A*_t (target)": [
            record.a_star.motivational_priority, record.a_star.decision_information_priority,
            record.a_star.intervention_readiness,
        ],
        "A_t (persisted)": (
            [record.a_t.motivational_priority, record.a_t.decision_information_priority,
             record.a_t.intervention_readiness]
            if record.a_t is not None else ["—", "—", "—"]
        ),
    }
    if record.state_delta is not None:
        rows["delta vs. prior turn"] = [
            record.state_delta["motivational_priority"],
            record.state_delta["decision_information_priority"],
            record.state_delta["intervention_readiness"],
        ]
    st.table(rows)


def _render_policy_panel(record: TurnRecord) -> None:
    st.markdown("**Policy (P_t)**")
    p_t = record.policy
    secondary = p_t.secondary.value if p_t.secondary else "—"
    st.table(
        {
            "field": [
                "primary", "secondary", "triggered_rules", "hard_constraints", "rationale_code",
                "state_could_influence_policy", "state_did_influence_policy",
            ],
            "value": [
                str(v) for v in (
                    p_t.primary.value, secondary, ", ".join(p_t.triggered_rules) or "—",
                    ", ".join(p_t.hard_constraints) or "—", p_t.rationale_code.value,
                    p_t.state_could_influence_policy, p_t.state_did_influence_policy,
                )
            ],
        }
    )


def _render_status_bar(record: TurnRecord) -> None:
    st.markdown("**System status**")
    st.table(
        {
            "field": [
                "estimator_status", "generator_status", "estimator_latency_ms", "generator_latency_ms",
                "model_id", "config_hash",
            ],
            "value": [
                str(v) for v in (
                    record.estimator_status.value, record.generator_status.value,
                    round(record.estimator_latency_ms, 2), round(record.generator_latency_ms, 2),
                    record.model_id, record.config_hash[:12] + "…",
                )
            ],
        }
    )


def render_replay_comparison(original: TurnRecord | None, replayed: TurnRecord) -> None:
    """Renders a replay/counterfactual result (fix for Phase 7 review blocker
    #3: "Render replay/counterfactual state and policy results in the
    dashboard. State/policy should be the default replay presentation.
    Response regeneration should be separately optional.").

    Pipeline.replay_turn() (frozen Phase 5 code, services/pipeline.py) always
    re-invokes the generator — it has no "skip regeneration" mode, and that
    cannot be changed here. So "optional" is implemented at THIS display
    layer instead: state (A*_t/A_t) and policy (P_t) are rendered directly,
    exactly like a live turn's own panels (reusing the same
    _render_agent_state_panel()/_render_policy_panel() helpers above so the
    two views are visually identical and comparable), while the regenerated
    response TEXT — which replay_turn() always produces as a side effect —
    is shown only inside a collapsed-by-default expander a researcher must
    choose to open.

    `original` may be None (app.py's _find_original_record() can fail to
    locate the pre-replay record, e.g. if session state was reset between
    recording and replay) — in that case only the replayed side renders,
    with a caption explaining why the comparison is one-sided.
    """
    st.subheader(f"Replay comparison — turn {replayed.turn_id}, {replayed.condition.value}")

    if original is None:
        st.caption(
            "Original turn record not found in this session's history — showing the "
            "replayed/counterfactual result only."
        )
        st.markdown(f"**Replayed** (rho={replayed.rho})")
        _render_agent_state_panel(replayed)
        st.divider()
        _render_policy_panel(replayed)
    else:
        col_orig, col_replay = st.columns(2)
        with col_orig:
            st.markdown(f"**Original** (rho={original.rho})")
            _render_agent_state_panel(original)
            st.divider()
            _render_policy_panel(original)
        with col_replay:
            st.markdown(f"**Replayed** (rho={replayed.rho})")
            _render_agent_state_panel(replayed)
            st.divider()
            _render_policy_panel(replayed)

    with st.expander("Regenerated response text (optional)", expanded=False):
        if original is not None:
            st.write("**Original:**", original.response_text)
        st.write("**Replayed:**", replayed.response_text)


def render(latest_comparison: dict[Condition, TurnRecord] | None, conditions: list[Condition]) -> None:
    st.subheader("Researcher dashboard")
    if not latest_comparison:
        st.caption("No turn recorded yet.")
        return

    tabs = st.tabs([c.value for c in conditions]) if conditions else []
    for tab, condition in zip(tabs, conditions):
        record = latest_comparison.get(condition)
        with tab:
            if record is None:
                st.caption("(not run this turn)")
                continue
            _render_human_goals_panel(record)
            st.divider()
            _render_agent_state_panel(record)
            st.divider()
            _render_policy_panel(record)
            st.divider()
            _render_status_bar(record)
