"""tests/test_researcher_dashboard.py — ui/researcher_dashboard.py's
render() and render_replay_comparison().

Originally added at the user's own explicit request, after they reported
(via a live Streamlit run) that TASK_FOCUSED/CURRENT_CUE/DYNAMIC appeared to
show "the same values" in the researcher dashboard: "Fix researcher_
dashboard so each condition tab renders the TurnRecord belonging to that
exact condition for the currently selected turn... Never reuse one
'latest_record' across all three tabs." Investigation at the time found no
bug (render()'s per-tab loop already keyed off the correct condition) but
this file's regression guard was added anyway, using hand-constructed
records so the assertion never depends on demo_fixture.yaml's own numbers
staying the same.

Rewritten for the user's own 15-item frontend-redesign spec (see /root/
.claude/plans/validated-cooking-wolf.md): render() no longer builds one
st.tabs() per condition at all — items 6/7 replaced the three repeated
per-tab Human/Goals + Agent-State + Policy panels with (a) ONE shared H_t/
c_t/G_t/D_t block, rendered once, and (b) ONE unified condition-by-column
"Condition-specific agent mechanism" table. The assertions below target
that new structure directly rather than per-tab panels, while still proving
the exact thing the user originally asked for: each condition's own column
in the unified table reflects THAT condition's own TurnRecord, never a
value borrowed from another condition or a single shared "latest_record".
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _dashboard_with_deliberately_different_condition_records():
    """Runs as an isolated Streamlit script (AppTest.from_function() below
    re-executes this function's own source in isolation) — every import and
    every value it needs must be self-contained in the function body, none
    of it can close over anything from the enclosing test module.

    Builds three TurnRecords for the SAME turn_id/comparison_id that are
    deliberately different per the user's own exact spec:
      - TASK_FOCUSED: a_star/a_t both None (never computed, §6.5), policy=INFORM
      - CURRENT_CUE:  rho=0.0,  a_info=0.30 (a distinct value), policy=INFORM
      - DYNAMIC:      rho=0.35, a_info=0.70 (a DIFFERENT value), policy=REDIRECT
    H_t (appraisal), G_t (goal_state), and D_t (derived_state) are the SAME
    object, shared across all three — exactly as Pipeline.compare() shares
    them by construction (§6.7) — so this test can also confirm that
    sharing survives rendering as ONE block, not that the per-condition
    fields differ.
    """
    from datetime import datetime, timezone

    from models.agent_state import AgentState, AgentTargetState
    from models.derived_state import DerivedInteractionState
    from models.enums import Condition, EstimatorStatus, EvidenceStrength, GeneratorStatus, Policy, RationaleCode
    from models.goal_state import GoalState
    from models.human_state import HumanAppraisal
    from models.observation import Observation
    from models.policy import PolicyState
    from models.turn_record import TurnRecord
    from ui import researcher_dashboard

    shared_g_t = GoalState(
        objective="decide whether to proceed",
        value_priorities={"autonomy": 0.6, "safety": 0.4},
        stakes=0.5, task_constraints={}, autonomy_weight=0.5, safety_risk=0.2,
    )
    shared_h_t = HumanAppraisal(
        goal_relevance=0.5, goal_congruence=0.0, uncertainty=0.5, perceived_control=0.5,
        agency=0.5, affect_intensity=0.3, possible_affect="test marker", evidence_tags=[],
        evidence_strength=EvidenceStrength.WEAK_INDIRECT,
    )
    shared_d_t = DerivedInteractionState(
        goal_conflict=0.1, evidence_ambiguity=0.1, goal_conflict_source="task_rule", ambiguity_source="task_rule",
    )
    shared_o_t = Observation(
        turn_id=5, timestamp=datetime.now(timezone.utc), user_text="hi", task_context={}, raw_history=[],
    )

    def _record(*, condition, a_t, a_star, rho, policy, did_influence):
        p_t = PolicyState(
            primary=policy,
            rationale_code=RationaleCode.REDIRECT_ELIGIBLE if policy is Policy.REDIRECT else RationaleCode.INFORM_NEEDED,
            state_could_influence_policy=condition is not Condition.TASK_FOCUSED,
            state_did_influence_policy=did_influence,
        )
        return TurnRecord(
            record_id=f"rec-{condition.value}", comparison_id="cmp-1", run_id="run-1",
            turn_id=5, timestamp=shared_o_t.timestamp, condition=condition,
            model_id="test-model", config_hash="deadbeef",
            outcome_baseline_value_priorities={"autonomy": 0.6, "safety": 0.4},
            observation=shared_o_t, appraisal=shared_h_t,
            evidence_strength=EvidenceStrength.WEAK_INDIRECT, c_t=0.5,
            goal_state=shared_g_t, derived_state=shared_d_t,
            a_prev=None, a_star=a_star, a_t=a_t, rho=rho, state_delta=None,
            policy=p_t, response_text=f"response for {condition.value}",
            estimator_status=EstimatorStatus.OK, generator_status=GeneratorStatus.OK,
            estimator_latency_ms=1.0, generator_latency_ms=1.0,
        )

    task_focused = _record(
        condition=Condition.TASK_FOCUSED, a_t=None, a_star=None, rho=0.0,
        policy=Policy.INFORM, did_influence=False,
    )
    current_cue = _record(
        condition=Condition.CURRENT_CUE,
        a_star=AgentTargetState(motivational_priority=0.5, decision_information_priority=0.30, intervention_readiness=0.4),
        a_t=AgentState(motivational_priority=0.5, decision_information_priority=0.30, intervention_readiness=0.4),
        rho=0.0, policy=Policy.INFORM, did_influence=False,
    )
    dynamic = _record(
        condition=Condition.DYNAMIC,
        a_star=AgentTargetState(motivational_priority=0.5, decision_information_priority=0.70, intervention_readiness=0.4),
        a_t=AgentState(motivational_priority=0.5, decision_information_priority=0.70, intervention_readiness=0.4),
        rho=0.35, policy=Policy.REDIRECT, did_influence=True,
    )

    comparison = {
        Condition.TASK_FOCUSED: task_focused,
        Condition.CURRENT_CUE: current_cue,
        Condition.DYNAMIC: dynamic,
    }
    researcher_dashboard.render(
        comparison, [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC],
    )


def _unified_mechanism_table(at):
    """Locates the one table with a "Dynamic" column — the new unified
    condition-comparison table (item 7) is the only st.table() call in
    render() whose columns are keyed by CONDITION_LABELS values rather than
    a plain "value" column, so this is an unambiguous way to find it."""
    for tbl in at.table:
        if "Dynamic" in tbl.value:
            return tbl.value
    raise AssertionError("no unified condition-mechanism table found (expected a 'Dynamic' column)")


def test_no_per_condition_tabs_rendered_anymore():
    """Item 7: the old three per-condition tabs are gone — replaced by one
    unified table. A regression here means someone reintroduced a per-tab
    layout (or ui/trajectory_view.py's own tabs leaked into this dashboard-
    only driver, which would itself be a bug since trajectory_view is never
    imported by this script)."""
    at = AppTest.from_function(_dashboard_with_deliberately_different_condition_records, default_timeout=60)
    at.run()
    assert not at.exception, f"researcher_dashboard.render() raised: {at.exception}"
    assert len(at.tabs) == 0


def test_shared_appraisal_and_context_rendered_once_not_per_condition():
    """Item 6: H_t/c_t/G_t/D_t are computed once and shared across every
    condition (§6.7) — the redesigned dashboard must show them ONCE, not
    once per condition as the pre-redesign per-tab layout did."""
    at = AppTest.from_function(_dashboard_with_deliberately_different_condition_records, default_timeout=60)
    at.run()
    assert not at.exception

    shared_headers = [m for m in at.markdown if "Shared human appraisal & decision context" in m.value]
    assert len(shared_headers) == 1, f"expected exactly one shared block, found {len(shared_headers)}"

    # The shared H_t table itself: exactly one, with the known values from
    # shared_h_t above (2 decimal places — item 8's display formatting).
    h_t_tables = [tbl for tbl in at.table if "goal_relevance (h_rel)" in list(tbl.value.get("field", []))]
    assert len(h_t_tables) == 1
    h_t_values = h_t_tables[0].value["value"].tolist()
    assert h_t_values[0] == "0.50"  # goal_relevance, formatted to 2dp


def test_condition_specific_columns_use_the_correct_records_not_one_shared_latest_record():
    """The user's own original complaint, re-targeted at the new unified
    table: each condition's OWN column must reflect that condition's own
    TurnRecord — never a value borrowed from another condition or one
    shared 'latest_record'."""
    at = AppTest.from_function(_dashboard_with_deliberately_different_condition_records, default_timeout=60)
    at.run()
    assert not at.exception

    table = _unified_mechanism_table(at)
    fields = table["field"].tolist()

    def _row(column, label):
        return table[column].tolist()[fields.index(label)]

    # -- Policy: each column shows ITS OWN condition's primary policy.
    assert _row("Task Focused", "Primary policy") == "INFORM"
    assert _row("Current Cue", "Primary policy") == "INFORM"
    assert _row("Dynamic", "Primary policy") == "REDIRECT"

    # -- TASK_FOCUSED never computes A*_t/A_t (§6.5) — "—" for every A*/A/
    # rho row, not merely a different number.
    assert _row("Task Focused", "Temporal persistence (rho)") == "—"
    assert _row("Task Focused", "Information priority (a_info) — target A*") == "—"
    assert _row("Task Focused", "Information priority (a_info) — A_t") == "—"

    # -- CURRENT_CUE: rho = 0, a_info = 0.30 (3dp per item 8).
    assert _row("Current Cue", "Temporal persistence (rho)") == "0.000"
    assert _row("Current Cue", "Information priority (a_info) — A_t") == "0.300"

    # -- DYNAMIC: rho = 0.35, a_info = 0.70 — genuinely different from
    # CURRENT_CUE's own value. This is the exact regression this test
    # guards against.
    assert _row("Dynamic", "Temporal persistence (rho)") == "0.350"
    assert _row("Dynamic", "Information priority (a_info) — A_t") == "0.700"
    assert _row("Current Cue", "Information priority (a_info) — A_t") != _row(
        "Dynamic", "Information priority (a_info) — A_t"
    )

    # -- state_could_influence_policy / state_did_influence_policy: TASK_
    # FOCUSED hard-set to False/False (§20.7); DYNAMIC's own record was
    # built with did_influence=True above.
    assert _row("Task Focused", "state_could_influence_policy") == "False"
    assert _row("Task Focused", "state_did_influence_policy") == "False"
    assert _row("Dynamic", "state_could_influence_policy") == "True"
    assert _row("Dynamic", "state_did_influence_policy") == "True"


def _replay_comparison_script():
    """Isolated script (see the dashboard driver's own note above for why
    every import/value must be self-contained here) driving
    render_replay_comparison() directly with two hand-built records: same
    condition/turn, different rho, deliberately different resulting policy
    (INFORM -> REDIRECT) so _replay_explanation()'s "policy changed" branch
    is exercised specifically (item 10)."""
    from datetime import datetime, timezone

    from models.agent_state import AgentState, AgentTargetState
    from models.derived_state import DerivedInteractionState
    from models.enums import Condition, EstimatorStatus, EvidenceStrength, GeneratorStatus, Policy, RationaleCode
    from models.goal_state import GoalState
    from models.human_state import HumanAppraisal
    from models.observation import Observation
    from models.policy import PolicyState
    from models.turn_record import TurnRecord
    from ui import researcher_dashboard

    g_t = GoalState(
        objective="decide whether to proceed", value_priorities={"autonomy": 0.6, "safety": 0.4},
        stakes=0.5, task_constraints={}, autonomy_weight=0.5, safety_risk=0.2,
    )
    h_t = HumanAppraisal(
        goal_relevance=0.5, goal_congruence=0.0, uncertainty=0.5, perceived_control=0.5,
        agency=0.5, affect_intensity=0.3, possible_affect=None, evidence_tags=[],
        evidence_strength=EvidenceStrength.WEAK_INDIRECT,
    )
    d_t = DerivedInteractionState(
        goal_conflict=0.1, evidence_ambiguity=0.1, goal_conflict_source="task_rule", ambiguity_source="task_rule",
    )
    o_t = Observation(turn_id=2, timestamp=datetime.now(timezone.utc), user_text="hi", task_context={}, raw_history=[])

    def _record(*, rho, a_info, policy, rationale, record_id):
        return TurnRecord(
            record_id=record_id, comparison_id="cmp-1", run_id="run-1",
            turn_id=2, timestamp=o_t.timestamp, condition=Condition.DYNAMIC,
            model_id="test-model", config_hash="deadbeef",
            outcome_baseline_value_priorities={"autonomy": 0.6, "safety": 0.4},
            observation=o_t, appraisal=h_t, evidence_strength=EvidenceStrength.WEAK_INDIRECT, c_t=0.65,
            goal_state=g_t, derived_state=d_t,
            a_prev=None,
            a_star=AgentTargetState(motivational_priority=0.5, decision_information_priority=a_info, intervention_readiness=0.65),
            a_t=AgentState(motivational_priority=0.5, decision_information_priority=a_info, intervention_readiness=0.65),
            rho=rho, state_delta=None,
            policy=PolicyState(
                primary=policy, rationale_code=rationale,
                state_could_influence_policy=True, state_did_influence_policy=policy is Policy.REDIRECT,
            ),
            response_text=f"response at rho={rho}",
            estimator_status=EstimatorStatus.OK, generator_status=GeneratorStatus.OK,
            estimator_latency_ms=1.0, generator_latency_ms=1.0,
        )

    original = _record(rho=0.0, a_info=0.40, policy=Policy.INFORM, rationale=RationaleCode.INFORM_NEEDED, record_id="rec-original")
    replayed = _record(rho=0.35, a_info=0.70, policy=Policy.REDIRECT, rationale=RationaleCode.REDIRECT_ELIGIBLE, record_id="rec-replayed")

    # Non-mutation guard (item 15: "counterfactual replay remains non-
    # mutating"): render_replay_comparison() is a pure display function — it
    # must not write back to either TurnRecord it's given. Snapshot both
    # before rendering and compare after.
    import streamlit as st

    st.session_state["_original_before"] = original.model_dump(mode="json")
    st.session_state["_replayed_before"] = replayed.model_dump(mode="json")
    researcher_dashboard.render_replay_comparison(original, replayed)
    st.session_state["_original_after"] = original.model_dump(mode="json")
    st.session_state["_replayed_after"] = replayed.model_dump(mode="json")


def test_render_replay_comparison_explains_the_policy_change_and_does_not_mutate_records():
    at = AppTest.from_function(_replay_comparison_script, default_timeout=60)
    at.run()
    assert not at.exception, f"render_replay_comparison() raised: {at.exception}"

    assert at.session_state["_original_before"] == at.session_state["_original_after"]
    assert at.session_state["_replayed_before"] == at.session_state["_replayed_after"]

    info_boxes = [box.value for box in at.info]
    assert any("changed the resulting policy" in msg for msg in info_boxes), info_boxes

    headers = [s.value for s in at.subheader]
    assert any(h.startswith("Counterfactual Analysis") for h in headers), headers


def _prepared_demo_turn2_script():
    """Runs the REAL config/demo_fixture.yaml scenario through the REAL
    Pipeline/ExperimentController (same construction app.py's own Offline
    backend uses — llm.adapter.OfflineDemoAdapter, services.demo_fixture.
    load_demo_fixture()) and renders BOTH turns through researcher_
    dashboard.render(), keeping only the SECOND turn's comparison — item
    15's explicit "prepared Turn 2 still produces CURRENT_CUE=INFORM and
    DYNAMIC=REDIRECT," verified as it actually appears in the dashboard's
    own unified table, not merely at the Pipeline layer (already covered
    analytically by tests/test_demo_fixture.py, which this test does not
    duplicate — this one is about the UI's own correct rendering of that
    same, already-verified result)."""
    from pathlib import Path

    import tempfile

    from llm.adapter import OfflineDemoAdapter
    from models.enums import Condition
    from services import demo_fixture as demo_fixture_module
    from services.demo_fixture import load_demo_fixture
    from services.experiment_controller import ExperimentController
    from services.goal_state_manager import GoalStateManager
    from services.state_manager import SessionState
    from tests.pipeline_fixtures import build_pipeline
    from ui import researcher_dashboard

    # AppTest.from_function() re-execs this function's own SOURCE from a
    # temp file, so a plain `Path(__file__)` here would resolve against
    # that temp file, not this real repo — services.demo_fixture is
    # imported normally (not source-execed), so ITS __file__ is a reliable
    # anchor back to the real repo root (same trick tests/pipeline_fixtures.
    # py's own CONFIG_PATH uses, one level up, for the exact same reason).
    fixture_path = Path(demo_fixture_module.__file__).resolve().parent.parent / "config" / "demo_fixture.yaml"
    fixture = load_demo_fixture(fixture_path)
    raw_appraisal_by_turn_id = {turn.turn_id: dict(turn.raw_appraisal) for turn in fixture.turns}
    adapter = OfflineDemoAdapter(raw_appraisal_by_turn_id)

    with tempfile.TemporaryDirectory() as tmp:
        pipeline, _adapter, _logger = build_pipeline(Path(tmp), adapter=adapter)
        controller = ExperimentController(pipeline, rho_dynamic=0.35)
        controller.elicit_outcome_baseline(fixture.outcome_baseline_value_priorities)
        session = SessionState(controller, GoalStateManager(), fixture.goal_state)

        all_conditions = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]
        for turn in fixture.turns:
            results = controller.compare(turn.user_text, turn.task_context, fixture.goal_state, all_conditions)
            session.record_comparison(results)

    researcher_dashboard.render(session.latest_comparison(), all_conditions)


def test_prepared_demo_turn2_dashboard_shows_current_cue_inform_dynamic_redirect():
    at = AppTest.from_function(_prepared_demo_turn2_script, default_timeout=60)
    at.run()
    assert not at.exception, f"prepared-demo dashboard render raised: {at.exception}"

    table = _unified_mechanism_table(at)
    fields = table["field"].tolist()
    primary_idx = fields.index("Primary policy")
    secondary_idx = fields.index("Secondary policy")
    assert table["Current Cue"].tolist()[primary_idx] == "INFORM"
    assert table["Current Cue"].tolist()[secondary_idx] == "ACKNOWLEDGE"
    assert table["Dynamic"].tolist()[primary_idx] == "REDIRECT"
    assert table["Dynamic"].tolist()[secondary_idx] == "ACKNOWLEDGE"


def _trajectory_script():
    """Isolated Streamlit script (see the dashboard driver's own note above
    for why every import/value must be self-contained here) — builds a real
    Pipeline/ExperimentController/SessionState via tests/pipeline_fixtures.
    py's own build_pipeline() helper (untouched by this investigation) and
    runs one real compare() through it, then renders trajectory_view exactly
    as app.py's own main() does."""
    import tempfile
    from pathlib import Path

    from models.enums import Condition
    from services.experiment_controller import ExperimentController
    from services.goal_state_manager import GoalStateManager
    from services.state_manager import SessionState
    from tests.pipeline_fixtures import build_pipeline, default_goal_state, default_raw_appraisal
    from ui import trajectory_view

    with tempfile.TemporaryDirectory() as tmp:
        pipeline, _adapter, _logger = build_pipeline(
            Path(tmp), extract_responses=[default_raw_appraisal()], generate_responses=["a", "b", "c"],
        )
        controller = ExperimentController(pipeline, rho_dynamic=0.35)
        controller.elicit_outcome_baseline({"autonomy": 0.6, "safety": 0.4})
        session = SessionState(controller, GoalStateManager(), default_goal_state())
        results = controller.compare(
            "hi", {}, default_goal_state(),
            [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC],
        )
        session.record_comparison(results)

    trajectory_view.render(session, [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC])


def test_trajectory_view_unaffected_by_this_change():
    """Explicit regression guard (user's own requirement: "Trajectory
    behavior must remain unchanged" — still true of this redesign: item 11
    only ADDS a combined Current-Cue-vs-Dynamic chart above the existing
    per-condition tabs; the tabs themselves are untouched). Tab LABELS now
    use the friendly ui/condition_labels.py names (item 4) rather than the
    raw enum value — this is the one visible change this test accounts for;
    everything else about trajectory_view's own render() is unaffected by
    anything in this file or in researcher_dashboard.py."""
    at = AppTest.from_function(_trajectory_script, default_timeout=60)
    at.run()
    assert not at.exception, f"trajectory_view.render() raised: {at.exception}"
    assert [t.label for t in at.tabs] == ["Task Focused", "Current Cue", "Dynamic"]
