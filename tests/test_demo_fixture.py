"""tests/test_demo_fixture.py — config/demo_fixture.yaml + services/
demo_fixture.py (§19.9/§20.15). Phase 6 gate file (blueprint §10 Phase 6):
"demo_fixture.yaml: multi-option/focal_option_id scenario tuned to the
analytical check, wired through compare()."

Named to match the three tests the user explicitly asked this phase to
implement: test_demo_turn2_satisfies_redirect_eligibility_and_flips_policy,
test_demo_fixture_repeatable, test_focal_option_validation_fails_fast. Plus
two more added in the post-delivery review round:
test_load_demo_fixture_fails_fast_on_invalid_focal_option and
test_load_demo_fixture_rejects_invalid_appraisal, covering the
load_demo_fixture()-level fix below.

Per the user's own explicit instruction (see services/demo_fixture.py's
module docstring and README.md's Phase 6 notes for the full record): the
frozen specification pins down an ANALYTICAL CONTRACT for this scenario
(numeric gates), not a literal one (option names/wording/numbers). This
file's job is therefore to compute the actual D_t/A*_t/A_t/policy THROUGH
the real Phase 1-5 code for the real fixture file and verify the flip — not
to assert against numbers copied out of the YAML by hand.

Post-delivery fix (user review of the first Phase 6 delivery): the first
draft of load_demo_fixture() parsed a YAML file into a DemoFixture without
ever calling validate_demo_fixture() itself, and stored each turn's
appraisal as a bare, unvalidated dict — so a malformed demo_fixture.yaml
(an unknown focal_option_id, or an out-of-range appraisal field like
affect_intensity=5.0) would load "successfully" and only fail later, mid-
demo, inside AppraisalEstimator's own live-LLM retry/fallback path — the
wrong failure mode for a fixture meant to be the reliable, pre-validated
offline interview fallback. load_demo_fixture() now validates both
unconditionally, at load time — see services/demo_fixture.py's own updated
docstring for the full fix.
"""

import dataclasses
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from models.enums import Condition, Policy
from services.derived_features import ScenarioConfigError
from services.demo_fixture import DemoFixture, DemoTurn, load_demo_fixture, run_demo_fixture, validate_demo_fixture
from services.experiment_controller import ExperimentController

from tests.pipeline_fixtures import build_pipeline

DEMO_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "config" / "demo_fixture.yaml"
ALL_CONDITIONS = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]


def _load_raw_fixture_dict() -> dict:
    """The parsed YAML dict underneath the real demo_fixture.yaml, before
    load_demo_fixture()'s own model construction — a starting point for
    tests that need a deliberately-broken on-disk fixture file, so those
    tests exercise load_demo_fixture() itself (the actual load-time
    boundary), not just validate_demo_fixture() called manually afterward."""
    with open(DEMO_FIXTURE_PATH) as f:
        return yaml.safe_load(f)


def _write_fixture_yaml(tmp_path: Path, raw: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path

# Frozen analytical-contract gates (§19.9/§20.15, reproduced from the user's
# own message — see this module's docstring). These numbers are the actual
# binding spec for Phase 6; everything else about the scenario is authored
# fixture content.
REDIRECT_H_INT_MIN = 0.65
REDIRECT_A_INTV_MIN = 0.60
REDIRECT_A_INFO_MIN = 0.55
REDIRECT_C_T_MIN = 0.60


def _build_controller(tmp_path, fixture: DemoFixture, conditions: list[Condition]) -> ExperimentController:
    """Wires one ExperimentController with a MockLLMAdapter scripted to
    "replay" this fixture's frozen H_t values in turn order — the mechanism
    by which a frozen demo fixture stands in for a live LLM call (see
    services/demo_fixture.py's DemoTurn.raw_appraisal docstring). extract_
    responses gets exactly one entry per turn (compare() calls the estimator
    once per turn, shared across every condition — see services/pipeline.py's
    compare()); generate_responses needs one entry per (turn, condition)
    pair, since compare() calls the generator once per condition."""
    extract_responses = [dict(turn.raw_appraisal) for turn in fixture.turns]
    generate_responses = [
        f"demo response turn={turn.turn_id}" for turn in fixture.turns for _ in conditions
    ]
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=extract_responses, generate_responses=generate_responses,
    )
    return ExperimentController(pipeline, rho_dynamic=0.35)


def test_demo_turn2_satisfies_redirect_eligibility_and_flips_policy(tmp_path):
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
    validate_demo_fixture(fixture)  # must not raise — the scenario is internally consistent

    controller = _build_controller(tmp_path, fixture, ALL_CONDITIONS)
    turn1_results, turn2_results = run_demo_fixture(fixture, controller, ALL_CONDITIONS)

    # --- Turn 1 gate: must establish a HIGH state (a*_intv >= 0.65) ---
    # A_1 == A*_1 exactly at the first affect-enabled turn of a run
    # (LinearPersistenceTransition.apply, is_first_affect_enabled_turn=True),
    # so CURRENT_CUE's and DYNAMIC's turn-1 a_star are identical and either
    # one checks this gate.
    turn1_a_star = turn1_results[Condition.CURRENT_CUE].a_star
    assert turn1_a_star.intervention_readiness >= 0.65
    assert turn1_results[Condition.DYNAMIC].a_star.intervention_readiness >= 0.65

    # --- Turn 2 gate: c_t, h_ctrl, h_int, h_unc, d_amb, d_goal, safety_risk ---
    # Read off any turn-2 record with a real H_t/D_t/c_t (identical across
    # conditions except TASK_FOCUSED, which never computes A*_t but still
    # gets a real H_t/D_t/c_t from compare()'s shared estimator call).
    dynamic_turn2 = turn2_results[Condition.DYNAMIC]
    h2, d2, c2, g2 = dynamic_turn2.appraisal, dynamic_turn2.derived_state, dynamic_turn2.c_t, dynamic_turn2.goal_state
    assert c2 >= REDIRECT_C_T_MIN
    assert h2.perceived_control >= 0.40
    assert h2.affect_intensity >= REDIRECT_H_INT_MIN
    assert h2.uncertainty < 0.70
    assert d2.evidence_ambiguity < 0.65
    assert d2.goal_conflict < 0.65
    assert g2.safety_risk < 0.70

    # --- CURRENT_CUE must fail at least one A-dependent REDIRECT threshold
    #     at Turn 2 (its A_2 IS A*_2 exactly, since CURRENT_CUE always runs
    #     at rho=0) ---
    cue_turn2 = turn2_results[Condition.CURRENT_CUE]
    assert cue_turn2.a_t.model_dump() == cue_turn2.a_star.model_dump()  # rho=0 collapse
    redirect_eligible_cue = (
        cue_turn2.a_star.decision_information_priority >= REDIRECT_A_INFO_MIN
        and cue_turn2.a_star.intervention_readiness >= REDIRECT_A_INTV_MIN
    )
    assert not redirect_eligible_cue

    # --- DYNAMIC at rho=0.35 must clear BOTH a_info and a_intv thresholds
    #     and select REDIRECT. Checked against a_t (the rho=0.35-blended
    #     PERSISTED state), not a_star — a_star_shared is the SAME A*_2
    #     target CURRENT_CUE also read above (compare() computes one shared
    #     A*_t per turn, §6.7); it is DYNAMIC's own persistence blend, not
    #     its target, that is engineered to clear these thresholds. ---
    assert dynamic_turn2.a_star.model_dump() == cue_turn2.a_star.model_dump()  # confirms the shared-target design
    assert dynamic_turn2.a_t.decision_information_priority >= REDIRECT_A_INFO_MIN
    assert dynamic_turn2.a_t.intervention_readiness >= REDIRECT_A_INTV_MIN
    assert dynamic_turn2.policy.primary == Policy.REDIRECT
    assert "R_REDIRECT" in dynamic_turn2.policy.triggered_rules

    # --- The actual flip: same turn 2 input, genuinely different policy ---
    assert cue_turn2.policy.primary != Policy.REDIRECT

    # --- Causal instrumentation: DYNAMIC's persistence blend is what
    #     crossed the line the A*_2-alone counterfactual would not have.
    #     CURRENT_CUE's own A_2 IS A*_2, so its counterfactual is a no-op:
    #     could=True (some A_t value could in principle influence policy)
    #     but did=False (this run's own A_t changed nothing). ---
    assert dynamic_turn2.policy.state_could_influence_policy is True
    assert dynamic_turn2.policy.state_did_influence_policy is True
    assert cue_turn2.policy.state_could_influence_policy is True
    assert cue_turn2.policy.state_did_influence_policy is False

    # --- Structural sanity: TASK_FOCUSED never computes A*_t/A_t at all ---
    tf_turn2 = turn2_results[Condition.TASK_FOCUSED]
    assert tf_turn2.a_star is None
    assert tf_turn2.a_t is None
    assert tf_turn2.policy.state_could_influence_policy is False
    assert tf_turn2.policy.state_did_influence_policy is False


def test_demo_fixture_repeatable(tmp_path):
    """The fixture is fully deterministic — no randomness anywhere in
    Phase 1-5's code (LinearPersistenceTransition, DerivedFeatureService,
    PolicyEngine are all pure functions of their inputs; H_t is scripted,
    not sampled). Running it twice, through two entirely independent
    Pipeline/ExperimentController instances (fresh run_id, fresh state),
    must produce numerically identical D_t/A*_t/A_t/policy output at every
    turn and condition."""
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)

    controller_a = _build_controller(tmp_path / "a", fixture, ALL_CONDITIONS)
    controller_b = _build_controller(tmp_path / "b", fixture, ALL_CONDITIONS)

    results_a = run_demo_fixture(fixture, controller_a, ALL_CONDITIONS)
    results_b = run_demo_fixture(fixture, controller_b, ALL_CONDITIONS)

    assert len(results_a) == len(results_b) == len(fixture.turns)
    for turn_results_a, turn_results_b in zip(results_a, results_b):
        assert set(turn_results_a) == set(turn_results_b) == set(ALL_CONDITIONS)
        for condition in ALL_CONDITIONS:
            record_a, record_b = turn_results_a[condition], turn_results_b[condition]
            assert record_a.derived_state == record_b.derived_state
            assert record_a.c_t == record_b.c_t
            assert record_a.appraisal == record_b.appraisal
            assert (record_a.a_star is None) == (record_b.a_star is None)
            if record_a.a_star is not None:
                assert record_a.a_star.model_dump() == record_b.a_star.model_dump()
                assert record_a.a_t.model_dump() == record_b.a_t.model_dump()
            assert record_a.policy.primary == record_b.policy.primary
            assert record_a.policy.secondary == record_b.policy.secondary
            assert record_a.policy.triggered_rules == record_b.policy.triggered_rules
            assert record_a.policy.state_could_influence_policy == record_b.policy.state_could_influence_policy
            assert record_a.policy.state_did_influence_policy == record_b.policy.state_did_influence_policy


def test_focal_option_validation_fails_fast():
    """validate_demo_fixture() must catch an internally-inconsistent
    task_context (here: a focal_option_id naming an option that doesn't
    exist) via the same ScenarioConfigError derive_interaction_state itself
    raises mid-demo — per that exception's own docstring's explicit promise
    for this fixture (see services/derived_features.py). "Fails fast" means
    this must raise from validate_demo_fixture() alone, with no
    ExperimentController/Pipeline/MockLLMAdapter ever constructed — this
    test builds none of those. Exercises validate_demo_fixture() directly
    against a deliberately-broken IN-PROCESS DemoFixture (no on-disk YAML
    needed for this one); see test_load_demo_fixture_fails_fast_on_invalid_
    focal_option below for the same failure mode exercised through
    load_demo_fixture() itself, i.e. the actual on-disk load-time boundary
    a researcher would hit."""
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
    real_turn = fixture.turns[0]

    broken_context = dict(real_turn.task_context)
    broken_context["focal_option_id"] = "not_a_real_option_id"
    broken_turn = dataclasses.replace(real_turn, task_context=broken_context)
    broken_fixture = dataclasses.replace(fixture, turns=(broken_turn,))

    with pytest.raises(ScenarioConfigError, match="not_a_real_option_id"):
        validate_demo_fixture(broken_fixture)


def test_load_demo_fixture_fails_fast_on_invalid_focal_option(tmp_path):
    """Post-delivery fix (user review): load_demo_fixture() previously
    parsed a YAML file into a DemoFixture and returned it WITHOUT ever
    calling validate_demo_fixture() itself — so a broken on-disk fixture
    file "loaded" successfully, and only failed later, mid-demo. This
    asserts the actual load-time boundary a researcher preparing a fixture
    file would hit: load_demo_fixture() on a broken FILE, not
    validate_demo_fixture() called manually on an in-process object (that
    is test_focal_option_validation_fails_fast above)."""
    raw = _load_raw_fixture_dict()
    raw["turns"][0]["focal_option_id"] = "not_a_real_option_id"
    broken_path = _write_fixture_yaml(tmp_path, raw, "broken_focal_option.yaml")

    with pytest.raises(ScenarioConfigError, match="not_a_real_option_id"):
        load_demo_fixture(broken_path)


def test_load_demo_fixture_rejects_invalid_appraisal(tmp_path):
    """Post-delivery fix (user review): load_demo_fixture() previously
    stored each turn's appraisal as a bare, unvalidated dict
    (raw_appraisal=dict(turn_raw["appraisal"])) — so an out-of-range value
    like affect_intensity=5.0 would load "successfully" here and only be
    rejected later by AppraisalEstimator._validate(), during a live-feeling
    demo run, via its retry-then-fallback path. That is the right failure
    mode for a genuinely flaky live LLM; it is the wrong one for a fixture
    that is supposed to be pre-validated once, offline, before an interview
    starts. load_demo_fixture() now round-trips every appraisal through a
    real HumanAppraisal(**...) construction at load time, so this raises
    pydantic's ValidationError immediately instead."""
    raw = _load_raw_fixture_dict()
    raw["turns"][0]["appraisal"]["affect_intensity"] = 5.0  # outside HumanAppraisal's [0,1] range
    broken_path = _write_fixture_yaml(tmp_path, raw, "broken_appraisal.yaml")

    with pytest.raises(ValidationError):
        load_demo_fixture(broken_path)


def test_load_demo_fixture_matches_frozen_analytical_contract():
    """A direct check of the frozen contract's non-numeric-gate clauses
    (option count, stakes range, autonomy_weight, safety_risk,
    no_undue_influence) against the parsed fixture — cheap insurance against
    a future edit to demo_fixture.yaml silently drifting outside the
    contract this file's own header comment documents."""
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)

    assert 2 <= len(fixture.turns[0].task_context["options"]) <= 3
    assert 0.75 <= fixture.goal_state.stakes <= 0.85
    assert fixture.goal_state.autonomy_weight == 0.90
    assert fixture.goal_state.safety_risk == 0.20
    assert fixture.goal_state.no_undue_influence is True
    assert len(fixture.turns) == 2
    for turn in fixture.turns:
        assert turn.task_context["required_evidence"]  # every turn has evidence fields
