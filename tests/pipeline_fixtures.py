"""Shared construction helpers for test_conditions.py, test_replay.py,
test_experiment_controller.py and test_outcome_baseline.py — all four
exercise the same fully-wired Pipeline/ExperimentController stack, so one
set of builders keeps default fixture values from drifting across files
(same rationale as tests/policy_fixtures.py, Phase 2). Not itself a test
module (no test_ prefix).
"""

from pathlib import Path

import yaml

from llm.adapter import MockLLMAdapter
from llm.fallback import FallbackTemplates
from models.goal_state import GoalState
from services.appraisal_estimator import AppraisalEstimator, confidence_map_from_config
from services.goal_state_manager import GoalStateManager
from services.logger import TurnLogger
from services.observation_builder import ObservationBuilder
from services.outcome_baseline import OutcomeBaselineStore
from services.pipeline import Pipeline, compute_config_hash
from services.policy_engine import PolicyConfig, PolicyEngine, build_affect_rules
from services.policy_engine_task_focused import build_task_focused_rules
from services.response_generator import ResponseGenerator
from services.state_transition import LinearPersistenceTransition

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_default_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_pipeline(
    tmp_path,
    *,
    model_id: str = "mock-model-v0",
    generate_responses: list = (),
    extract_responses: list = (),
    adapter=None,
):
    """Wires one Pipeline exactly the way a real caller (Phase 7's UI, or a
    test) would: one MockLLMAdapter serving both AppraisalEstimator and
    ResponseGenerator (llm.adapter.LLMAdapter is a single Protocol with
    both methods — nothing requires two separate adapter instances, and
    sharing one keeps every service reading the SAME config-derived values
    a real deployment would use), config-injected PolicyEngine/Transition
    exactly as Phase 1/2 built them, and a TurnLogger writing into
    tmp_path (pytest's own per-test scratch directory) rather than the
    real repo's data/logs/.

    Returns (pipeline, adapter, logger) — most tests only need pipeline,
    but adapter lets a test inspect exactly what was sent to the LLM, and
    logger lets a test inspect what was actually persisted without
    re-reading the JSONL file back off disk.
    """
    config = load_default_config()
    if adapter is None:
        adapter = MockLLMAdapter(
            responses=list(extract_responses), generate_responses=list(generate_responses)
        )

    observation_builder = ObservationBuilder()
    estimator = AppraisalEstimator(adapter, confidence_map_from_config(config))
    transition = LinearPersistenceTransition(config["transition"]["weights"])
    policy_cfg = PolicyConfig.from_mapping(config)
    policy_engine = PolicyEngine(policy_cfg, build_affect_rules(policy_cfg), build_task_focused_rules(policy_cfg))
    generator = ResponseGenerator(adapter, FallbackTemplates())
    goal_manager = GoalStateManager()
    logger = TurnLogger(Path(tmp_path) / "turns.jsonl")
    outcome_baseline_store = OutcomeBaselineStore()

    pipeline = Pipeline(
        observation_builder=observation_builder,
        estimator=estimator,
        transition=transition,
        policy_engine=policy_engine,
        generator=generator,
        goal_manager=goal_manager,
        logger=logger,
        outcome_baseline_store=outcome_baseline_store,
        model_id=model_id,
        config_hash=compute_config_hash(config),
    )
    return pipeline, adapter, logger


def default_goal_state(**overrides) -> GoalState:
    defaults = dict(
        objective="decide whether to proceed",
        value_priorities={"autonomy": 0.6, "safety": 0.4},
        stakes=0.5,
        task_constraints={},
        autonomy_weight=0.5,
        safety_risk=0.1,
    )
    defaults.update(overrides)
    return GoalState(**defaults)


def default_raw_appraisal(**overrides) -> dict:
    """A well-formed raw extraction: exactly HumanAppraisal's own field
    names (§5.2), matching tests/test_appraisal_estimator.py's own _raw()."""
    defaults = dict(
        goal_relevance=0.5,
        goal_congruence=0.0,
        uncertainty=0.4,
        perceived_control=0.6,
        agency=0.5,
        affect_intensity=0.3,
        possible_affect="mild concern",
        evidence_tags=["explicit_statement"],
        evidence_strength="EXPLICIT",
    )
    defaults.update(overrides)
    return defaults


DEFAULT_OUTCOME_BASELINE_PRIORITIES = {"autonomy": 0.6, "safety": 0.4}
