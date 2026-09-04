"""tests/test_appraisal_estimator.py — M2 (AppraisalEstimator), blueprint
§6.2/§7.1/§20.3. Phase 3 gate (blueprint §10 Phase 3): "Estimator returns
valid H_t/c_t on the mock fixture; RETRY_OK and fallback paths both tested."

Also covers three rows the blueprint's own §9 test map assigns to
test_schemas.py by name (test_estimator_status_retry_ok_vs_ok,
test_fallback_appraisal_exact_values, test_confidence_mapping_deterministic)
— filed here instead, alongside the module they actually exercise, since
that is where this project has consistently put phase-introduced service
tests (Phase 1/2 did the same for rows nominally tied to earlier files).
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from llm.adapter import MockLLMAdapter
from models.enums import EstimatorStatus, EvidenceStrength
from models.goal_state import GoalState
from models.observation import Observation
from services.appraisal_estimator import AppraisalEstimator, confidence_map_from_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _load_confidence_map() -> dict[EvidenceStrength, float]:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return confidence_map_from_config(config)


def _goal_state(**overrides) -> GoalState:
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


def _observation(**overrides) -> Observation:
    defaults = dict(turn_id=1, timestamp=datetime.now(timezone.utc), user_text="I think option A is fine.")
    defaults.update(overrides)
    return Observation(**defaults)


def _raw(**overrides) -> dict:
    """A well-formed raw extraction: exactly HumanAppraisal's own field
    names (§5.2), per llm/adapter.py's documented schema decision."""
    defaults = dict(
        goal_relevance=0.5,
        goal_congruence=0.0,
        uncertainty=0.4,
        perceived_control=0.6,
        agency=0.5,
        affect_intensity=0.3,
        possible_affect="mild frustration",
        evidence_tags=["explicit_statement"],
        evidence_strength="EXPLICIT",
    )
    defaults.update(overrides)
    return defaults


@pytest.fixture(scope="module")
def confidence_map() -> dict[EvidenceStrength, float]:
    return _load_confidence_map()


def test_confidence_map_from_config_matches_yaml_exactly(confidence_map):
    with open(CONFIG_PATH) as f:
        raw_yaml = yaml.safe_load(f)["confidence_map"]
    assert confidence_map == {EvidenceStrength(k): v for k, v in raw_yaml.items()}
    assert confidence_map[EvidenceStrength.EXPLICIT] == pytest.approx(0.85)
    assert confidence_map[EvidenceStrength.INSUFFICIENT] == pytest.approx(0.30)


# ---------- clean first-pass success ----------


def test_estimate_returns_ok_status_on_clean_first_pass(confidence_map):
    adapter = MockLLMAdapter(responses=[_raw(evidence_strength="STRONG_INDIRECT")])
    estimator = AppraisalEstimator(adapter, confidence_map)
    o_t, g_t = _observation(), _goal_state()

    h_t, evidence_strength, c_t, status = estimator.estimate(o_t, g_t)

    assert status == EstimatorStatus.OK
    assert evidence_strength == EvidenceStrength.STRONG_INDIRECT
    assert c_t == pytest.approx(confidence_map[EvidenceStrength.STRONG_INDIRECT])
    assert h_t.goal_relevance == pytest.approx(0.5)
    assert len(adapter.calls) == 1
    assert adapter.calls[0] == (o_t, g_t, False)


# ---------- retry path: RETRY_OK ----------


@pytest.mark.parametrize(
    "bad_raw",
    [
        None,                                           # malformed/unparseable JSON, or a transport failure
        "not a dict",                                    # non-dict raw
        {**_raw(), "goal_relevance": 5.0},                # out-of-range field (Pydantic rejects)
        {k: v for k, v in _raw().items() if k != "uncertainty"},   # missing required field
        {**_raw(), "evidence_strength": "VERY_SURE"},     # not a real EvidenceStrength member
    ],
)
def test_estimate_returns_retry_ok_status_when_first_response_invalid(confidence_map, bad_raw):
    adapter = MockLLMAdapter(responses=[bad_raw, _raw(evidence_strength="WEAK_INDIRECT")])
    estimator = AppraisalEstimator(adapter, confidence_map)
    o_t, g_t = _observation(), _goal_state()

    h_t, evidence_strength, c_t, status = estimator.estimate(o_t, g_t)

    assert status == EstimatorStatus.RETRY_OK
    assert evidence_strength == EvidenceStrength.WEAK_INDIRECT
    assert c_t == pytest.approx(confidence_map[EvidenceStrength.WEAK_INDIRECT])
    assert len(adapter.calls) == 2
    assert adapter.calls[0] == (o_t, g_t, False)
    assert adapter.calls[1] == (o_t, g_t, True)   # the repair retry


def test_estimator_status_retry_ok_vs_ok(confidence_map):
    """Named to match the blueprint's §9 test map row exactly: RETRY_OK vs
    OK is not cosmetic — a clean first pass and a successful-after-repair
    pass must be distinguishable in the returned status, not just inferred
    from call count."""
    clean_adapter = MockLLMAdapter(responses=[_raw()])
    clean_status = AppraisalEstimator(clean_adapter, confidence_map).estimate(_observation(), _goal_state())[3]

    repaired_adapter = MockLLMAdapter(responses=[None, _raw()])
    repaired_status = AppraisalEstimator(repaired_adapter, confidence_map).estimate(_observation(), _goal_state())[3]

    assert clean_status == EstimatorStatus.OK
    assert repaired_status == EstimatorStatus.RETRY_OK
    assert clean_status != repaired_status


# ---------- fallback path: both attempts invalid ----------


def test_estimate_returns_fallback_when_both_attempts_invalid(confidence_map):
    adapter = MockLLMAdapter(responses=[None, "still not usable"])
    estimator = AppraisalEstimator(adapter, confidence_map)
    o_t, g_t = _observation(), _goal_state()

    h_t, evidence_strength, c_t, status = estimator.estimate(o_t, g_t)

    assert status == EstimatorStatus.FALLBACK_LOW_CONFIDENCE
    assert evidence_strength == EvidenceStrength.INSUFFICIENT
    assert c_t == pytest.approx(confidence_map[EvidenceStrength.INSUFFICIENT])
    assert h_t == AppraisalEstimator.FALLBACK_APPRAISAL
    assert len(adapter.calls) == 2   # exactly one retry, never a third attempt


def test_fallback_appraisal_exact_values():
    """Named to match the blueprint's §9 test map row exactly: the fallback
    is a deterministic constant, not recomputed per failure."""
    fallback = AppraisalEstimator.FALLBACK_APPRAISAL
    assert fallback.goal_relevance == pytest.approx(0.0)
    assert fallback.goal_congruence == pytest.approx(0.0)
    assert fallback.uncertainty == pytest.approx(1.0)
    assert fallback.perceived_control == pytest.approx(0.5)
    assert fallback.agency == pytest.approx(0.5)
    assert fallback.affect_intensity == pytest.approx(0.0)
    assert fallback.possible_affect is None
    assert fallback.evidence_tags == ["ESTIMATOR_FAILURE"]
    assert fallback.evidence_strength == EvidenceStrength.INSUFFICIENT


def test_fallback_returns_independent_equal_instances(confidence_map):
    """A class-level constant template, never recomputed (blueprint note,
    §6.2) — but each call must get its OWN instance, not a shared reference
    to the template itself (implementation-review fix, see estimate()'s
    comment: HumanAppraisal is a plain mutable BaseModel, so handing out the
    same object would let one turn's mutation of its own H_t corrupt the
    fallback for every later turn)."""
    adapter1 = MockLLMAdapter(responses=[None, None])
    adapter2 = MockLLMAdapter(responses=[None, None])
    h1 = AppraisalEstimator(adapter1, confidence_map).estimate(_observation(), _goal_state())[0]
    h2 = AppraisalEstimator(adapter2, confidence_map).estimate(_observation(), _goal_state())[0]
    assert h1 == AppraisalEstimator.FALLBACK_APPRAISAL
    assert h2 == AppraisalEstimator.FALLBACK_APPRAISAL
    assert h1 is not h2
    assert h1 is not AppraisalEstimator.FALLBACK_APPRAISAL


def test_mutating_returned_fallback_does_not_change_template(confidence_map):
    adapter = MockLLMAdapter(responses=[None, None])
    h_t = AppraisalEstimator(adapter, confidence_map).estimate(_observation(), _goal_state())[0]

    h_t.uncertainty = 0.2

    assert AppraisalEstimator.FALLBACK_APPRAISAL.uncertainty == pytest.approx(1.0)


# ---------- c_t is never a function of H_t's own field values ----------


def test_confidence_mapping_deterministic(confidence_map):
    """Named to match the blueprint's §9 test map row exactly: for every
    EvidenceStrength category, c_t equals confidence_map[evidence_strength]
    exactly."""
    for evidence_strength in EvidenceStrength:
        adapter = MockLLMAdapter(responses=[_raw(evidence_strength=evidence_strength.value)])
        _, returned_strength, c_t, _ = AppraisalEstimator(adapter, confidence_map).estimate(
            _observation(), _goal_state()
        )
        assert returned_strength == evidence_strength
        assert c_t == pytest.approx(confidence_map[evidence_strength])


def test_confidence_mapping_is_never_a_function_of_h_t_fields(confidence_map):
    """Same evidence_strength, wildly different H_t field values -> identical
    c_t (§19.1: H_t and c_t are parallel outputs, c_t never derived from
    H_t's own fields)."""
    adapter_low = MockLLMAdapter(
        responses=[_raw(evidence_strength="CONTRADICTORY", goal_relevance=0.0, affect_intensity=0.0, uncertainty=0.0)]
    )
    adapter_high = MockLLMAdapter(
        responses=[_raw(evidence_strength="CONTRADICTORY", goal_relevance=1.0, affect_intensity=1.0, uncertainty=1.0)]
    )
    _, _, c_t_low, _ = AppraisalEstimator(adapter_low, confidence_map).estimate(_observation(), _goal_state())
    _, _, c_t_high, _ = AppraisalEstimator(adapter_high, confidence_map).estimate(_observation(), _goal_state())
    assert c_t_low == pytest.approx(c_t_high)
    assert c_t_low == pytest.approx(confidence_map[EvidenceStrength.CONTRADICTORY])


# ---------- possible_affect / evidence_tags pass through, optional fields ----------


def test_optional_fields_default_when_absent_from_raw(confidence_map):
    raw = _raw()
    del raw["possible_affect"]
    del raw["evidence_tags"]
    adapter = MockLLMAdapter(responses=[raw])
    h_t = AppraisalEstimator(adapter, confidence_map).estimate(_observation(), _goal_state())[0]
    assert h_t.possible_affect is None
    assert h_t.evidence_tags == []
