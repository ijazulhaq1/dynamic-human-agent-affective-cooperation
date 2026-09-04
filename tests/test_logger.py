"""tests/test_logger.py — TurnLogger, blueprint §6.8. Not itself a Phase 5
gate file (see tests/test_goal_state_manager.py's own note — same
convention) but written per this project's one-test-file-per-module
practice.
"""

from datetime import datetime, timezone

from models.derived_state import DerivedInteractionState
from models.enums import Condition, EstimatorStatus, EvidenceStrength, GeneratorStatus, Policy, RationaleCode
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from models.observation import Observation
from models.policy import PolicyState
from models.turn_record import TurnRecord
from services.logger import TurnLogger


def _turn_record(**overrides) -> TurnRecord:
    defaults = dict(
        record_id="rec-1",
        comparison_id=None,
        run_id="run-1",
        turn_id=1,
        timestamp=datetime.now(timezone.utc),
        condition=Condition.DYNAMIC,
        model_id="mock-model-v0",
        config_hash="deadbeef",
        outcome_baseline_value_priorities={"autonomy": 0.6, "safety": 0.4},
        observation=Observation(turn_id=1, timestamp=datetime.now(timezone.utc), user_text="hello"),
        appraisal=HumanAppraisal(
            goal_relevance=0.5, goal_congruence=0.0, uncertainty=0.4, perceived_control=0.6,
            agency=0.5, affect_intensity=0.3, evidence_strength=EvidenceStrength.EXPLICIT,
        ),
        evidence_strength=EvidenceStrength.EXPLICIT,
        c_t=0.85,
        goal_state=GoalState(
            objective="decide", value_priorities={"autonomy": 0.6, "safety": 0.4}, stakes=0.5,
            task_constraints={}, autonomy_weight=0.5, safety_risk=0.1,
        ),
        derived_state=DerivedInteractionState(
            goal_conflict=0.1, evidence_ambiguity=0.1,
            goal_conflict_source="task_rule", ambiguity_source="task_rule",
        ),
        rho=0.0,
        policy=PolicyState(
            primary=Policy.INFORM, rationale_code=RationaleCode.MINIMAL_SUPPORT,
            state_could_influence_policy=False, state_did_influence_policy=False,
        ),
        response_text="Here's some information.",
        estimator_status=EstimatorStatus.OK,
        generator_status=GeneratorStatus.OK,
        estimator_latency_ms=1.0,
        generator_latency_ms=1.0,
    )
    defaults.update(overrides)
    return TurnRecord(**defaults)


def test_persist_appends_jsonl_line(tmp_path):
    logger = TurnLogger(tmp_path / "turns.jsonl")
    record = _turn_record()
    logger.persist(record)

    lines = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = TurnRecord.model_validate_json(lines[0])
    assert parsed.record_id == record.record_id


def test_persist_is_append_only(tmp_path):
    logger = TurnLogger(tmp_path / "turns.jsonl")
    logger.persist(_turn_record(record_id="rec-1"))
    logger.persist(_turn_record(record_id="rec-2"))

    lines = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert TurnRecord.model_validate_json(lines[0]).record_id == "rec-1"
    assert TurnRecord.model_validate_json(lines[1]).record_id == "rec-2"


def test_persist_keeps_in_memory_records(tmp_path):
    logger = TurnLogger(tmp_path / "turns.jsonl")
    r1, r2 = _turn_record(record_id="rec-1"), _turn_record(record_id="rec-2")
    logger.persist(r1)
    logger.persist(r2)
    assert logger.records == [r1, r2]


def test_persist_creates_parent_directory(tmp_path):
    nested = tmp_path / "nested" / "dir" / "turns.jsonl"
    logger = TurnLogger(nested)
    logger.persist(_turn_record())
    assert nested.exists()
