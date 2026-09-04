"""app.py — Dynamic Affective Cooperation, live researcher rehearsal UI.
Blueprint §3 repo layout (app.py at the repository root) / §10 Phase 7
("ui/* (Streamlit), real llm/adapter.py backend"). Run with:

    streamlit run app.py

Phase 7's own gate (§10): "Full live rehearsal × 10 (§16.2, §20.17);
offline/replay fallback verified with network disabled." Both halves of
that gate map directly onto this app's two backend modes, in the sidebar:

  - "Anthropic (live)" — AnthropicLLMAdapter (llm/adapter.py), a real,
    network-calling backend. This is the "Full live rehearsal × 10" path —
    a researcher runs this app with a real ANTHROPIC_API_KEY and rehearses
    (at least) ten live turns end to end.
  - "Mock (offline/replay)" — OfflineDemoAdapter (llm/adapter.py), driven
    by config/demo_fixture.yaml's own frozen H_t values (services/
    demo_fixture.py, Phase 6). This is the "offline/replay fallback
    verified with network disabled" path — running this mode makes zero
    network calls, so a researcher can literally disconnect the network and
    confirm the demo scenario still runs end to end deterministically, any
    number of times, including replaying past turns after the demo has
    already run (see _init_mock_session()'s own docstring below for why
    this is OfflineDemoAdapter and not MockLLMAdapter, the Phase 3 test
    double used everywhere else in this codebase's own test suite).

Neither acceptance check is something this codebase can complete on its
own: §16.2/§20.17's ten-live-turn rehearsal needs a real API key and a
human interviewer/participant exchange, and "network disabled" needs a
human to actually disconnect the network and watch nothing break. Building
everything needed to PERFORM both checks — and getting the automatable
part (the offline/replay path, which needs no credential and no human
judgment about response quality) actually working end to end — is this
file's job; performing them is the researcher's own final acceptance step,
same as this project has already flagged for every other manually-gated
build-order line (see README.md's Phase 7 section for the full account).

Judgment call (documented gap, flagged for review): the blueprint gives
app.py no body at all beyond naming it in the repo layout. Everything below
— page layout, session bootstrap, the demo-scenario/live-turn split, how
config_hash is computed from ALL FOUR runtime yaml files (the Phase 5/6 fix
round's own "REMINDER for Phase 6/7 integration" — services/pipeline.py's
compute_config_hash() docstring — finally acted on here) — is authored for
this prototype.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import yaml

from llm.adapter import AnthropicLLMAdapter, LLMAdapter, OfflineDemoAdapter
from llm.fallback import FallbackTemplates
from models.enums import Condition
from models.goal_state import GoalState
from models.turn_record import TurnRecord
from services.appraisal_estimator import AppraisalEstimator, confidence_map_from_config
from services.demo_fixture import load_demo_fixture
from services.experiment_controller import ExperimentController
from services.goal_state_manager import GoalStateManager
from services.logger import TurnLogger
from services.observation_builder import ObservationBuilder
from services.outcome_baseline import OutcomeBaselineStore
from services.pipeline import Pipeline, compute_config_hash
from services.policy_engine import PolicyConfig, PolicyEngine, build_affect_rules
from services.policy_engine_task_focused import build_task_focused_rules
from services.response_generator import ResponseGenerator
from services.state_manager import SessionState
from services.state_transition import LinearPersistenceTransition

from ui import experiment_controls, interaction_view, researcher_dashboard, trajectory_view

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"
DEMO_FIXTURE_PATH = CONFIG_DIR / "demo_fixture.yaml"
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "logs" / "turns.jsonl"
RUNTIME_CONFIG_FILES = ("default.yaml", "conditions.yaml", "policy_rules.yaml", "demo_fixture.yaml")
ALL_CONDITIONS = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]

MOCK_BACKEND = "Mock (offline/replay)"
LIVE_BACKEND = "Anthropic (live)"


def _load_full_runtime_config() -> dict:
    """Merges every runtime YAML file into one dict before hashing — the
    exact "Phase 6/7 integration" reminder services/pipeline.py's own
    compute_config_hash() docstring left for this phase: default.yaml +
    conditions.yaml + policy_rules.yaml + demo_fixture.yaml, so a change to
    ANY one of them changes the resulting config_hash. Phase 0-6's own
    tests deliberately hashed default.yaml alone (sufficient for their own
    narrower scope, per that docstring) — this is the first real caller
    that needs the complete picture."""
    merged: dict = {}
    for name in RUNTIME_CONFIG_FILES:
        with open(CONFIG_DIR / name) as f:
            merged[name] = yaml.safe_load(f)
    return merged


def _build_pipeline(llm_adapter: LLMAdapter, model_id: str) -> tuple[Pipeline, float]:
    """Wires one live Pipeline exactly the way tests/pipeline_fixtures.py's
    build_pipeline() wires a test one (same construction, same config
    source) — except config_hash is computed over ALL FOUR runtime files
    (see _load_full_runtime_config above), and TurnLogger writes to this
    repo's own data/logs/ directory (§3's repo layout) instead of a pytest
    tmp_path. Returns (pipeline, rho_dynamic) — the caller still owns
    constructing ExperimentController(pipeline, rho_dynamic=...) itself, so
    this function's own responsibility stays "build the Pipeline," matching
    every other builder in this codebase being a single-purpose class/
    function."""
    full_config = _load_full_runtime_config()
    default_config = full_config["default.yaml"]

    observation_builder = ObservationBuilder()
    estimator = AppraisalEstimator(llm_adapter, confidence_map_from_config(default_config))
    transition = LinearPersistenceTransition(default_config["transition"]["weights"])
    policy_cfg = PolicyConfig.from_mapping(default_config)
    policy_engine = PolicyEngine(policy_cfg, build_affect_rules(policy_cfg), build_task_focused_rules(policy_cfg))
    generator = ResponseGenerator(llm_adapter, FallbackTemplates())
    goal_manager = GoalStateManager()
    DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = TurnLogger(DEFAULT_LOG_PATH)
    outcome_baseline_store = OutcomeBaselineStore()

    pipeline = Pipeline(
        observation_builder=observation_builder, estimator=estimator, transition=transition,
        policy_engine=policy_engine, generator=generator, goal_manager=goal_manager, logger=logger,
        outcome_baseline_store=outcome_baseline_store, model_id=model_id,
        config_hash=compute_config_hash(full_config),
    )
    return pipeline, default_config["transition"]["rho_dynamic"]


def _default_goal_state() -> GoalState:
    """The frozen demo scenario's own G_t (config/demo_fixture.yaml,
    Phase 6) doubles as this app's default starting goal_state — a
    reasonable, analytically-verified starting point for either backend
    mode, rather than authoring a second, unrelated default scenario just
    for the live-turn path."""
    return load_demo_fixture(DEMO_FIXTURE_PATH).goal_state


def _init_mock_session() -> SessionState:
    """Mock (offline/replay) backend: OfflineDemoAdapter (llm/adapter.py) —
    NOT MockLLMAdapter (Phase 3's test double) — replays config/
    demo_fixture.yaml's own frozen H_t values by turn_id, unlimited number
    of times. Post-delivery fix (user review): MockLLMAdapter's queues are
    ONE-SHOT by design (a valued safety net in tests/), so a scripted queue
    sized for exactly one demo run would raise "called more times than
    scripted" the moment a researcher replayed a turn, or re-ran the demo
    without resetting — a live crash on exactly the path that is supposed
    to be the RELIABLE offline fallback. See OfflineDemoAdapter's own
    docstring for the full account; tests/test_app_smoke.py's
    test_replay_after_full_demo_does_not_raise and
    test_demo_scenario_runs_again_cleanly_after_reset both exercise this."""
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
    raw_appraisal_by_turn_id = {turn.turn_id: dict(turn.raw_appraisal) for turn in fixture.turns}
    adapter = OfflineDemoAdapter(raw_appraisal_by_turn_id)
    pipeline, rho_dynamic = _build_pipeline(adapter, model_id="mock-offline-replay")
    controller = ExperimentController(pipeline, rho_dynamic=rho_dynamic)
    controller.elicit_outcome_baseline(fixture.outcome_baseline_value_priorities)
    return SessionState(controller, GoalStateManager(), fixture.goal_state)


def _init_live_session(api_key: str | None) -> SessionState:
    """Anthropic (live) backend — a real, network-calling adapter.
    api_key=None lets anthropic.Anthropic() fall back to its own documented
    ANTHROPIC_API_KEY environment-variable lookup; a key typed into the
    sidebar (when the environment variable is absent) is passed through
    explicitly instead."""
    import anthropic  # local import: only this live-backend path needs the real package installed

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    adapter = AnthropicLLMAdapter(client)
    pipeline, rho_dynamic = _build_pipeline(adapter, model_id=adapter.model)
    controller = ExperimentController(pipeline, rho_dynamic=rho_dynamic)
    goal_state = _default_goal_state()
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
    controller.elicit_outcome_baseline(fixture.outcome_baseline_value_priorities)
    return SessionState(controller, GoalStateManager(), goal_state)


def _find_original_record(session: SessionState, turn_id: int, condition: Condition) -> TurnRecord | None:
    """Looks up the ORIGINAL (non-replayed) TurnRecord a replay request was
    made against, from session.comparisons — needed so the replay
    comparison view (researcher_dashboard.render_replay_comparison, called
    from main() below) can show original-vs-replayed side by side. Returns
    None if no matching record is found (a caller replaying a turn_id/
    condition combination that was never actually compared — Pipeline.
    replay_turn() itself would already have raised KeyError before this is
    ever called in that case, so None here is defensive, not an expected
    path)."""
    for records in session.comparisons:
        record = records.get(condition)
        if record is not None and record.turn_id == turn_id:
            return record
    return None


def main() -> None:
    st.set_page_config(page_title="Dynamic Affective Cooperation — Live Rehearsal", layout="wide")
    st.title("Dynamic Affective Cooperation — researcher rehearsal")
    st.caption(
        "Lund University Cognitive Science / Agentic AI Research Group — interview prototype. "
        "See README.md for what is and is not frozen by the specification."
    )

    env_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    with st.sidebar:
        st.header("LLM backend")
        backend = st.radio(
            "Backend", [MOCK_BACKEND, LIVE_BACKEND], index=1 if env_key_present else 0,
            help=(
                "Mock replays config/demo_fixture.yaml's frozen, analytically-verified scenario "
                "with NO network calls — Phase 7's 'offline/replay fallback verified with network "
                "disabled' check. Anthropic makes real API calls for live free-text rehearsal — "
                "Phase 7's 'full live rehearsal × 10' check."
            ),
        )
        api_key_input = None
        if backend == LIVE_BACKEND and not env_key_present:
            api_key_input = st.text_input("ANTHROPIC_API_KEY", type="password")
            if not api_key_input:
                st.warning("Enter an API key to use the live backend, or switch to Mock.")

    if st.session_state.get("backend") != backend or "session" not in st.session_state:
        try:
            if backend == MOCK_BACKEND:
                st.session_state.session = _init_mock_session()
            else:
                if not (api_key_input or env_key_present):
                    st.stop()
                st.session_state.session = _init_live_session(api_key_input)
            st.session_state.backend = backend
        except ImportError as exc:
            st.error(str(exc))
            st.stop()

    session: SessionState = st.session_state.session

    controls = experiment_controls.render(session)

    if controls.reset_requested:
        # A brand new SessionState (not session.reset() on the existing
        # one) is deliberate here, not merely simpler: Mock mode's adapter
        # has a scripted, ONE-SHOT extract/generate queue (see
        # _init_mock_session's own docstring) — after a demo run consumes
        # it, session.reset() alone would leave a fresh run_id pointed at
        # an already-empty queue, so the next "Run demo_fixture.yaml" click
        # would hit MockLLMAdapter's own "called more times than scripted"
        # AssertionError. Discarding st.session_state["session"] and
        # ["backend"] and letting the normal init branch below rebuild
        # everything from scratch (a fresh adapter with a freshly reloaded
        # queue, a fresh ExperimentController, a fresh SessionState) is the
        # one reset path that is correct for BOTH backends — the Anthropic
        # client rebuilt for the live case has no equivalent one-shot state
        # to lose, so unifying on this path costs it nothing but
        # reconstructing one lightweight client object.
        del st.session_state["backend"]
        del st.session_state["session"]
        st.session_state.pop("last_replay", None)
        st.rerun()

    if controls.replay_request is not None:
        turn_id, condition, replay_rho = controls.replay_request
        try:
            replayed = session.controller.replay_turn(turn_id, condition, replay_rho)
        except KeyError as exc:
            st.error(str(exc))
        else:
            original = _find_original_record(session, turn_id, condition)
            st.session_state["last_replay"] = (original, replayed)

    if backend == MOCK_BACKEND:
        st.subheader("Offline/replay: the frozen demo scenario")
        st.caption(
            "Runs config/demo_fixture.yaml's two turns through every condition, with NO network call "
            "— the exact scenario tests/test_demo_fixture.py verifies analytically in Phase 6."
        )
        if st.button("Run demo_fixture.yaml (both turns, all three conditions)"):
            fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
            for turn in fixture.turns:
                results = session.controller.compare(
                    turn.user_text, turn.task_context, fixture.goal_state, ALL_CONDITIONS,
                )
                session.record_comparison(results)
            st.session_state.pop("last_replay", None)  # a fresh run supersedes any earlier replay comparison
            st.rerun()
    else:
        st.subheader("Live turn")
        with st.form("live_turn_form", clear_on_submit=True):
            user_text = st.text_area("Participant message")
            submitted = st.form_submit_button("Send")
        if submitted and user_text.strip():
            results = session.controller.compare(
                user_text, {}, session.goal_state, controls.conditions, rho_override=controls.rho_override,
            )
            session.record_comparison(results)
            st.session_state.pop("last_replay", None)

    st.divider()

    last_replay = st.session_state.get("last_replay")
    if last_replay is not None:
        original, replayed = last_replay
        researcher_dashboard.render_replay_comparison(original, replayed)
        st.divider()

    latest = session.latest_comparison()
    interaction_view.render(latest, controls.conditions)
    researcher_dashboard.render(latest, controls.conditions)
    trajectory_view.render(session, controls.conditions)


if __name__ == "__main__":
    main()
