"""tests/test_schemas.py — §16.1 schema-rejection tests; Phase 0 gate (blueprint §9, §10 Phase 0).

Every test here checks a Pydantic model construction and nothing else: no
services, no estimator, no policy engine exist yet at this phase. Later
schema-adjacent rows in the blueprint's test map (confidence mapping, exact
fallback H_t values, retry status, fallback-path TurnRecord validity) belong
to the phases that introduce the modules they test (AppraisalEstimator,
Pipeline) and are added there, not here.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.agent_state import AgentState, AgentTargetState
from models.derived_state import DerivedInteractionState
from models.enums import (
    Condition,
    EstimatorStatus,
    EvidenceStrength,
    GeneratorStatus,
    GoalUpdateSource,
    Policy,
    RationaleCode,
)
from models.goal_state import GoalState, OutcomeBaseline
from models.human_state import HumanAppraisal
from models.observation import Observation, Turn
from models.policy import PolicyState
from models.turn_record import TurnRecord


# ---------- construction helpers (valid baselines, overridden per test) ----------


def _valid_goal_state(**overrides) -> GoalState:
    defaults = dict(
        objective="decide whether to proceed",
        value_priorities={"autonomy": 0.6, "safety": 0.4},
        stakes=0.5,
        task_constraints={},
        autonomy_weight=0.5,
        safety_risk=0.2,
    )
    defaults.update(overrides)
    return GoalState(**defaults)


def _valid_human_appraisal(**overrides) -> HumanAppraisal:
    defaults = dict(
        goal_relevance=0.5,
        goal_congruence=0.0,
        uncertainty=0.5,
        perceived_control=0.5,
        agency=0.5,
        affect_intensity=0.3,
        possible_affect=None,
        evidence_tags=[],
        evidence_strength=EvidenceStrength.WEAK_INDIRECT,
    )
    defaults.update(overrides)
    return HumanAppraisal(**defaults)


def _valid_observation(**overrides) -> Observation:
    defaults = dict(
        turn_id=1,
        timestamp=datetime.now(timezone.utc),
        user_text="hello",
        task_context={},
        raw_history=[],
    )
    defaults.update(overrides)
    return Observation(**defaults)


def _valid_derived_state(**overrides) -> DerivedInteractionState:
    defaults = dict(
        goal_conflict=0.1,
        evidence_ambiguity=0.1,
        goal_conflict_source="task_rule",
        ambiguity_source="task_rule",
    )
    defaults.update(overrides)
    return DerivedInteractionState(**defaults)


def _valid_policy_state(**overrides) -> PolicyState:
    defaults = dict(
        primary=Policy.INFORM,
        rationale_code=RationaleCode.MINIMAL_SUPPORT,
        state_could_influence_policy=False,
        state_did_influence_policy=False,
    )
    defaults.update(overrides)
    return PolicyState(**defaults)


def _turn_record_kwargs(**overrides) -> dict:
    kwargs = dict(
        record_id="rec-1",
        comparison_id=None,
        run_id="run-1",
        turn_id=1,
        timestamp=datetime.now(timezone.utc),
        condition=Condition.CURRENT_CUE,
        model_id="mock",
        config_hash="deadbeef",
        outcome_baseline_value_priorities={"autonomy": 0.6, "safety": 0.4},
        observation=_valid_observation(),
        appraisal=_valid_human_appraisal(),
        evidence_strength=EvidenceStrength.WEAK_INDIRECT,
        c_t=0.55,
        goal_state=_valid_goal_state(),
        derived_state=_valid_derived_state(),
        a_prev=None,
        a_star=None,
        a_t=AgentState(motivational_priority=0.4, decision_information_priority=0.4, intervention_readiness=0.3),
        rho=0.0,
        state_delta=None,
        policy=_valid_policy_state(),
        response_text="okay, one moment.",
        estimator_status=EstimatorStatus.OK,
        generator_status=GeneratorStatus.OK,
        estimator_latency_ms=10.0,
        generator_latency_ms=10.0,
    )
    kwargs.update(overrides)
    return kwargs


# ---------- out-of-range fields rejected ----------


@pytest.mark.parametrize(
    "field,value",
    [
        ("goal_relevance", 1.5),
        ("goal_relevance", -0.1),
        ("goal_congruence", 1.5),
        ("goal_congruence", -1.5),
        ("uncertainty", 2.0),
        ("perceived_control", -1.0),
        ("agency", 1.1),
        ("affect_intensity", -0.01),
    ],
)
def test_human_appraisal_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        _valid_human_appraisal(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [("stakes", 1.5), ("autonomy_weight", -0.2), ("safety_risk", 1.01)],
)
def test_goal_state_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        _valid_goal_state(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("motivational_priority", 1.2),
        ("decision_information_priority", -0.5),
        ("intervention_readiness", 1.001),
    ],
)
def test_agent_target_state_out_of_range_rejected(field, value):
    base = dict(motivational_priority=0.5, decision_information_priority=0.5, intervention_readiness=0.5)
    base[field] = value
    with pytest.raises(ValidationError):
        AgentTargetState(**base)


@pytest.mark.parametrize("field,value", [("goal_conflict", 1.5), ("evidence_ambiguity", -0.3)])
def test_derived_state_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        _valid_derived_state(**{field: value})


def test_observation_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _valid_observation(observed_confidence=1.5)


def test_turn_record_rho_out_of_range_rejected():
    with pytest.raises(ValidationError):
        TurnRecord(**_turn_record_kwargs(rho=1.5))


def test_turn_record_c_t_out_of_range_rejected():
    with pytest.raises(ValidationError):
        TurnRecord(**_turn_record_kwargs(c_t=-0.1))


# ---------- missing required fields rejected ----------


def test_human_appraisal_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        HumanAppraisal(goal_relevance=0.5)  # missing uncertainty, perceived_control, ... evidence_strength


def test_goal_state_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        GoalState(stakes=0.5)  # missing objective, autonomy_weight, safety_risk


def test_observation_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        Observation(user_text="hi")  # missing turn_id, timestamp


def test_policy_state_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        PolicyState(primary=Policy.INFORM)  # missing rationale_code, both state-relevance flags


def test_turn_record_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        TurnRecord(record_id="r1")  # missing nearly everything


# ---------- enum values match the frozen spec exactly ----------


def test_condition_enum_values():
    assert {c.value for c in Condition} == {"TASK_FOCUSED", "CURRENT_CUE", "DYNAMIC"}


def test_policy_enum_values():
    assert {p.value for p in Policy} == {"INFORM", "ACKNOWLEDGE", "CLARIFY", "CHALLENGE", "REDIRECT", "DEFER"}


def test_evidence_strength_enum_values():
    assert {e.value for e in EvidenceStrength} == {
        "EXPLICIT",
        "STRONG_INDIRECT",
        "WEAK_INDIRECT",
        "CONTRADICTORY",
        "INSUFFICIENT",
    }


def test_estimator_status_has_retry_ok_distinct_from_ok():
    # implementation-review fix: RETRY_OK must exist and be distinguishable from a clean first pass
    assert EstimatorStatus.RETRY_OK != EstimatorStatus.OK
    assert {s.value for s in EstimatorStatus} == {"OK", "RETRY_OK", "FALLBACK_LOW_CONFIDENCE", "ERROR"}


# ---------- GoalState.with_explicit_update is the only mutation path ----------


def test_goal_state_with_explicit_update_increments_version_and_source():
    g1 = _valid_goal_state(goal_version=1, update_source=GoalUpdateSource.RESEARCHER_CONFIG)
    g2 = g1.with_explicit_update(GoalUpdateSource.HUMAN_EXPLICIT, safety_risk=0.9)
    assert g2.goal_version == g1.goal_version + 1
    assert g2.update_source == GoalUpdateSource.HUMAN_EXPLICIT
    assert g2.safety_risk == 0.9
    assert g1.safety_risk != 0.9  # original untouched — with_explicit_update never mutates in place


def test_goal_state_with_explicit_update_revalidates_changed_fields():
    # implementation-review fix: model_copy(update=...) does NOT revalidate,
    # so this used to silently produce a GoalState with safety_risk=2.0.
    g = _valid_goal_state()
    with pytest.raises(ValidationError):
        g.with_explicit_update(GoalUpdateSource.HUMAN_EXPLICIT, safety_risk=2.0)


def test_goal_state_with_explicit_update_revalidates_value_priorities():
    g = _valid_goal_state()
    with pytest.raises(ValidationError):
        g.with_explicit_update(GoalUpdateSource.HUMAN_EXPLICIT, value_priorities={"autonomy": 4.0})


# ---------- value_priorities values are range-validated on every model that has them ----------


@pytest.mark.parametrize("bad_priorities", [{"autonomy": 4.0}, {"autonomy": -0.5}])
def test_goal_state_value_priorities_out_of_range_rejected(bad_priorities):
    with pytest.raises(ValidationError):
        _valid_goal_state(value_priorities=bad_priorities)


@pytest.mark.parametrize("bad_priorities", [{"safety": 4.0}, {"safety": -0.5}])
def test_outcome_baseline_value_priorities_out_of_range_rejected(bad_priorities):
    with pytest.raises(ValidationError):
        OutcomeBaseline(run_id="run-1", value_priorities=bad_priorities)


# ---------- OutcomeBaseline is frozen, deeply (Pydantic v2 ConfigDict + MappingProxyType) ----------


def test_outcome_baseline_field_reassignment_rejected():
    baseline = OutcomeBaseline(run_id="run-1", value_priorities={"safety": 1.0})
    with pytest.raises(ValidationError):
        baseline.value_priorities = {"safety": 0.0}


def test_outcome_baseline_nested_mutation_rejected():
    # implementation-review fix: frozen=True alone only blocks *reassigning*
    # the field — the dict it pointed to was still mutable in place. This is
    # the central methodological guarantee (the pre-interaction baseline
    # cannot change during the run), so it gets its own explicit test rather
    # than relying on the reassignment test above to imply it.
    baseline = OutcomeBaseline(run_id="run-1", value_priorities={"safety": 1.0})
    with pytest.raises(TypeError):
        baseline.value_priorities["safety"] = 0.0
    assert baseline.value_priorities["safety"] == 1.0  # confirm it truly didn't change


def test_outcome_baseline_serializes_to_plain_dict():
    # MappingProxyType must not leak into JSONL logging (a later phase) —
    # serialization should hand back an ordinary, JSON-safe dict.
    baseline = OutcomeBaseline(run_id="run-1", value_priorities={"safety": 1.0})
    dumped = baseline.model_dump()
    assert dumped["value_priorities"] == {"safety": 1.0}
    assert type(dumped["value_priorities"]) is dict
    assert baseline.model_dump_json()  # must not raise


# ---------- Turn is non-recursive; Observation.raw_history holds Turn, not Observation ----------


def test_turn_from_observation_does_not_embed_full_observation():
    obs = _valid_observation(task_context={"task_event": {"kind": "deadline_moved"}})
    turn = Turn.from_observation(obs)
    assert turn.turn_id == obs.turn_id
    assert turn.user_text == obs.user_text
    assert turn.task_event == {"kind": "deadline_moved"}
    # Turn has no field capable of holding a nested Observation or its
    # task_context — this is what keeps raw_history: list[Turn] non-recursive
    # structurally, not just by convention.
    assert not hasattr(turn, "task_context")
    assert not hasattr(turn, "raw_history")


def test_observation_raw_history_accepts_list_of_turn():
    prior = Turn(turn_id=1, user_text="earlier turn")
    obs = _valid_observation(turn_id=2, raw_history=[prior])
    assert obs.raw_history == [prior]


# ---------- TurnRecord: valid construction, unique-identity fields present ----------


def test_turn_record_valid_construction_round_trip():
    record = TurnRecord(**_turn_record_kwargs())
    assert record.record_id == "rec-1"
    assert record.comparison_id is None
    assert record.policy.primary is Policy.INFORM


def test_turn_record_comparison_id_optional_and_settable():
    record = TurnRecord(**_turn_record_kwargs(comparison_id="cmp-1"))
    assert record.comparison_id == "cmp-1"
