"""demo_fixture.py — loads, validates, and drives config/demo_fixture.yaml
(§19.9/§20.15) through the live pipeline. Blueprint §10 Phase 6: "demo_
fixture.yaml: multi-option/focal_option_id scenario tuned to the analytical
check, wired through compare()."

Judgment call, decided together with the user before this module was
written (see the conversation, not the blueprint): the recovered frozen
specification freezes an ANALYTICAL CONTRACT for this scenario — option
count, pre-elicited value priorities as a category, structured option-to-
value impacts as a mechanism, required-evidence fields, stakes in
[0.75, 0.85], autonomy_weight=0.90, safety_risk=0.20, no_undue_influence
always true, and (crucially) the exact numeric gates the two-turn
trajectory must clear: Turn 1 establishes a*_intv >= 0.65; Turn 2 must have
c_t >= 0.60, h_ctrl >= 0.40, h_int >= 0.65, h_unc < 0.70, d_amb < 0.65,
d_goal < 0.65, safety_risk < 0.70; CURRENT_CUE must fail at least one
A-dependent REDIRECT threshold at Turn 2; DYNAMIC at rho=0.35 must clear
a_info >= 0.55 and a_intv >= 0.60 and select REDIRECT. It does NOT freeze a
literal scenario — no exact option names, exact value_priorities numbers,
exact option-impact values, exact focal_option_id, or exact turn wording.
Those are authored, implementation-level fixture content here, engineered
(not guessed by trial and error — see the analytical derivation recorded
in config/demo_fixture.yaml's own header comment) to satisfy every one of
the frozen numeric gates above, verified by running this scenario through
the REAL Phase 1-5 code (derive_interaction_state, LinearPersistenceTransition,
PolicyEngine) rather than asserted by hand. Wording correction (user
review): the margins above the gates are NOT uniformly "clean" or
"healthy" — Turn 1's a*_intv margin is comfortable (+0.068 over 0.65), but
DYNAMIC's Turn-2 a_info/a_intv margins over the REDIRECT thresholds are
deliberately narrow (~+0.010 each). That narrowness is intentional, not a
defect: the scenario is engineered so DYNAMIC just barely crosses the line
CURRENT_CUE stays just under, and the whole trajectory is fully
deterministic (no randomness anywhere in Phase 1-5's code), so any positive
margin reproduces identically on every run — see
test_demo_fixture_repeatable. See config/demo_fixture.yaml's own header
comment for the exact numbers.

Data model: DemoTurn/DemoFixture are plain frozen dataclasses (same
rationale as Phase 2's Ctx/Rule and Phase 4's GenerationContract — pure
in-process containers, no JSON boundary or range validation of their own
beyond what the GoalState/HumanAppraisal construction calls inside
load_demo_fixture() enforce, plus the derive_interaction_state()-backed
consistency check validate_demo_fixture() runs, which load_demo_fixture()
now always applies before returning — see both functions' own docstrings.

Post-delivery fix (user review of the first Phase 6 delivery): this is
supposed to be the reliable, offline interview fallback, so a malformed
prepared fixture must fail LOADING the file, not fail only if a caller also
remembers to call validate_demo_fixture() afterward, and not fail only
later, mid-demo, inside AppraisalEstimator's own live-LLM retry/fallback
path (which is the wrong failure mode for a fixture that is supposed to be
pre-validated once, offline, before an interview starts). Two gaps fixed:
load_demo_fixture() now calls validate_demo_fixture() on the parsed fixture
before returning it (previously that was a second, easy-to-forget manual
step — the module docstring's own claim that construction "already
enforces" this was aspirational, not actually true, until this fix); and
every turn's raw_appraisal is now round-tripped through a real
HumanAppraisal(**...) construction at load time (previously stored as a
bare, unvalidated dict), so an out-of-range value like affect_intensity=5.0
raises pydantic's ValidationError at load time instead of silently loading
and only being rejected later by AppraisalEstimator._validate() during a
live-feeling demo run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from models.enums import Condition
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from models.observation import Observation
from models.turn_record import TurnRecord
from services.derived_features import derive_interaction_state
from services.experiment_controller import ExperimentController


@dataclass(frozen=True)
class DemoTurn:
    turn_id: int
    user_text: str
    task_context: dict[str, Any]      # options + focal_option_id + required_evidence, ready for compare()
    raw_appraisal: dict[str, Any]     # flat dict, HumanAppraisal's own field names — scriptable directly
    # into MockLLMAdapter's extract_appraisal responses queue, standing in for the live LLM call this
    # frozen fixture replaces (see this module's own docstring for why H_t — never D_t, which is a pure
    # function of task_context/goal_state and is recomputed live every run — is the one thing that needs
    # "replaying" rather than recomputing).


@dataclass(frozen=True)
class DemoFixture:
    goal_state: GoalState
    outcome_baseline_value_priorities: dict[str, float]
    turns: tuple[DemoTurn, ...]


def load_demo_fixture(path: str | Path) -> DemoFixture:
    """Parses config/demo_fixture.yaml into a DemoFixture, then validates it
    before returning — see this module's own docstring for the post-
    delivery fix that made both of the steps below unconditional rather
    than a manual second call a caller could forget:

    1. Each turn's raw appraisal is round-tripped through a real
       HumanAppraisal(**...) construction (then re-flattened via
       model_dump(mode="json") back into the same plain dict shape
       MockLLMAdapter's extract_appraisal queue and
       AppraisalEstimator._validate() both expect — see DemoTurn.
       raw_appraisal's own docstring). This raises pydantic's
       ValidationError immediately for an out-of-range or malformed
       appraisal field (e.g. affect_intensity=5.0), rather than silently
       loading a fixture that would only fail later, mid-demo, inside
       AppraisalEstimator's own live-LLM retry/fallback path — the wrong
       failure mode for a fixture that is supposed to be pre-validated
       once, offline.
    2. validate_demo_fixture(fixture) is called on the fully-parsed
       DemoFixture before it is returned, so a scenario-level
       inconsistency (e.g. an unknown focal_option_id) raises
       ScenarioConfigError at load time too — not only if a caller
       separately remembers to call validate_demo_fixture() afterward.

    validate_demo_fixture() itself stays a separate, public function (not
    inlined here) so test_focal_option_validation_fails_fast can still
    construct a deliberately-broken DemoFixture in-process and call it
    directly, without needing a second broken YAML file on disk."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    goal_state = GoalState(**raw["goal_state"])
    options = raw["options"]

    turns = []
    for turn_raw in raw["turns"]:
        task_context = {
            "options": options,
            "focal_option_id": turn_raw["focal_option_id"],
            "required_evidence": turn_raw["required_evidence"],
        }
        validated_appraisal = HumanAppraisal(**turn_raw["appraisal"])  # raises ValidationError on bad input
        turns.append(
            DemoTurn(
                turn_id=turn_raw["turn_id"],
                user_text=turn_raw["user_text"],
                task_context=task_context,
                raw_appraisal=validated_appraisal.model_dump(mode="json"),
            )
        )

    fixture = DemoFixture(
        goal_state=goal_state,
        outcome_baseline_value_priorities=dict(raw["outcome_baseline"]["value_priorities"]),
        turns=tuple(turns),
    )
    validate_demo_fixture(fixture)  # raises ScenarioConfigError on inconsistency, at load time
    return fixture


def validate_demo_fixture(fixture: DemoFixture) -> None:
    """Checked at fixture-load time, per services/derived_features.py's own
    ScenarioConfigError docstring: "for demo_fixture.yaml... checked at
    fixture-load time via validate_demo_fixture(), not discovered as a bare
    KeyError mid-demo." load_demo_fixture() now calls this itself,
    unconditionally, before returning (post-delivery fix — see this
    module's own docstring); it stays a separate public function, rather
    than being inlined into load_demo_fixture(), so a test can still build
    a deliberately-broken DemoFixture in-process and call this directly, no
    second broken YAML file required (see test_focal_option_validation_
    fails_fast in tests/test_demo_fixture.py).

    Judgment call (documented gap, flagged for review): the blueprint names
    this function and its purpose but never gives its body. Rather than
    re-implementing derive_interaction_state's own focal_option_id/options-
    vs-value_priorities consistency checks a second time — risking the two
    silently drifting apart — this constructs a throwaway Observation per
    turn and calls the REAL derive_interaction_state(o_t, g_t) eagerly for
    every turn in the fixture. If any turn's task_context is internally
    inconsistent (an unknown focal_option_id, an option impact naming a
    value G_t.value_priorities doesn't weight, an invalid required_evidence
    status), the exact same ScenarioConfigError derive_interaction_state
    would raise mid-demo is raised here instead, before any turn actually
    runs — this function's return value (D_t for each turn) is discarded;
    only the raising-or-not matters here."""
    for turn in fixture.turns:
        probe = Observation(
            turn_id=turn.turn_id,
            timestamp=datetime.now(timezone.utc),
            user_text=turn.user_text,
            task_context=turn.task_context,
        )
        derive_interaction_state(probe, fixture.goal_state)  # raises ScenarioConfigError on inconsistency


def run_demo_fixture(
    fixture: DemoFixture, controller: ExperimentController, conditions: list[Condition]
) -> list[dict[Condition, TurnRecord]]:
    """Drives every turn in the fixture through ExperimentController.
    compare(), in order — this IS the "wired through compare()" requirement
    from the blueprint's own Phase 6 build-order line. The offline/
    reproducible rehearsal path runs through the EXACT SAME
    ExperimentController.compare() every real demo turn uses; there is no
    separate code path built just for fixture replay. `controller` is
    expected to already be constructed (by the caller — see
    tests/test_demo_fixture.py) with an LLMAdapter whose extract_appraisal
    responses are scripted from fixture.turns[i].raw_appraisal, in order,
    so this function itself never touches the adapter directly. Elicits the
    fixture's own OutcomeBaseline once, before the first turn, exactly as
    OutcomeBaseline's own "elicited before turn 1" contract requires."""
    controller.elicit_outcome_baseline(fixture.outcome_baseline_value_priorities)
    return [
        controller.compare(turn.user_text, turn.task_context, fixture.goal_state, conditions)
        for turn in fixture.turns
    ]
