"""app.py — Streamlit interface for the Human–AI Dynamic Affective
Cooperation prototype.

Supports:
- a deterministic offline demonstration (no network calls),
- live OpenAI and Anthropic backends,
- comparison across the three experimental conditions,
- state/policy inspection via the researcher dashboard,
- counterfactual rho replay,
- run export as JSON.

Run with:

    streamlit run app.py

`config_hash` (services/pipeline.py's `compute_config_hash()`) is computed
only from the four runtime YAML files under config/ — no API key is ever
included in it, logged, or written to any `TurnRecord`. Provider/model
identity is tracked separately, in `TurnRecord.model_id`.

See README.md for the full build history and the manual acceptance steps
(a real multi-turn live rehearsal, network-disabled offline verification)
that this file supports but cannot complete on its own.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
import yaml

from llm.adapter import AnthropicLLMAdapter, LLMAdapter, OfflineDemoAdapter, OpenAILLMAdapter
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

OFFLINE_BACKEND = "Offline"
OPENAI_BACKEND = "OpenAI"
ANTHROPIC_BACKEND = "Anthropic"
BACKEND_OPTIONS = [OFFLINE_BACKEND, OPENAI_BACKEND, ANTHROPIC_BACKEND]


def _load_full_runtime_config() -> dict:
    """Merges every runtime YAML file (default/conditions/policy_rules/
    demo_fixture) into one dict before hashing, so a change to any one of
    them changes the resulting config_hash."""
    merged: dict = {}
    for name in RUNTIME_CONFIG_FILES:
        with open(CONFIG_DIR / name) as f:
            merged[name] = yaml.safe_load(f)
    return merged


def _build_pipeline(llm_adapter: LLMAdapter, model_id: str) -> tuple[Pipeline, float]:
    """Wires one live Pipeline the same way tests/pipeline_fixtures.py's
    build_pipeline() wires a test one, except config_hash covers all four
    runtime files (see _load_full_runtime_config above) and TurnLogger
    writes to this repo's data/logs/ directory instead of a pytest
    tmp_path. Returns (pipeline, rho_dynamic)."""
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
    """The frozen demo scenario's own G_t doubles as this app's default
    starting goal_state for a live session."""
    return load_demo_fixture(DEMO_FIXTURE_PATH).goal_state


def _init_mock_session() -> tuple[SessionState, str]:
    """Offline backend: OfflineDemoAdapter (llm/adapter.py) replays
    config/demo_fixture.yaml's frozen H_t values by turn_id, any number of
    times — running the demo again or replaying a past turn never exhausts
    it. Returns (session, model_id); model_id is a fixed, non-secret label
    ("offline-demo-fixture"), not a provider model string."""
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
    raw_appraisal_by_turn_id = {turn.turn_id: dict(turn.raw_appraisal) for turn in fixture.turns}
    adapter = OfflineDemoAdapter(raw_appraisal_by_turn_id)
    model_id = "offline-demo-fixture"
    pipeline, rho_dynamic = _build_pipeline(adapter, model_id=model_id)
    controller = ExperimentController(pipeline, rho_dynamic=rho_dynamic)
    controller.elicit_outcome_baseline(fixture.outcome_baseline_value_priorities)
    return SessionState(controller, GoalStateManager(), fixture.goal_state), model_id


def _init_live_session(api_key: str | None) -> tuple[SessionState, str]:
    """Anthropic backend — a real, network-calling adapter. api_key=None
    lets anthropic.Anthropic() fall back to its own ANTHROPIC_API_KEY
    environment-variable lookup; a key typed into the sidebar is passed
    through explicitly otherwise. Never stored beyond this local variable,
    never logged, never folded into config_hash."""
    import anthropic  # local import: only this live-backend path needs the real package installed

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    adapter = AnthropicLLMAdapter(client)
    pipeline, rho_dynamic = _build_pipeline(adapter, model_id=adapter.model)
    controller = ExperimentController(pipeline, rho_dynamic=rho_dynamic)
    goal_state = _default_goal_state()
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
    controller.elicit_outcome_baseline(fixture.outcome_baseline_value_priorities)
    return SessionState(controller, GoalStateManager(), goal_state), adapter.model


def _init_openai_session(api_key: str | None) -> tuple[SessionState, str]:
    """OpenAI backend, alongside Anthropic (never replacing it) — a real,
    network-calling adapter. Mirrors _init_live_session() above, calling
    the same _build_pipeline(): H_t -> D_t -> A*_t -> A_t -> P_t is built
    identically regardless of which live backend is selected; only the
    adapter object differs."""
    import openai  # local import: only this backend path needs the real package installed

    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    adapter = OpenAILLMAdapter(client)
    pipeline, rho_dynamic = _build_pipeline(adapter, model_id=adapter.model)
    controller = ExperimentController(pipeline, rho_dynamic=rho_dynamic)
    goal_state = _default_goal_state()
    fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
    controller.elicit_outcome_baseline(fixture.outcome_baseline_value_priorities)
    return SessionState(controller, GoalStateManager(), goal_state), adapter.model


def _find_original_record(session: SessionState, turn_id: int, condition: Condition) -> TurnRecord | None:
    """Looks up the original (pre-replay) TurnRecord a replay request was
    made against, so the replay view can show original vs. counterfactual
    side by side. Returns None if no matching record is found (defensive —
    Pipeline.replay_turn() itself would already have raised KeyError before
    this is ever called for a turn_id/condition that was never compared)."""
    for records in session.comparisons:
        record = records.get(condition)
        if record is not None and record.turn_id == turn_id:
            return record
    return None


def _render_export_button(session: SessionState) -> None:
    """Serializes session.comparisons/single_turn_records as JSON, via each
    TurnRecord's own model_dump(mode="json") — no new computation, nothing
    under models/ or services/ touched. Renders nothing at all until at
    least one turn has been recorded."""
    has_data = bool(session.comparisons or session.single_turn_records)
    if not has_data:
        return
    payload = {
        "run_id": session.run_id,
        "comparisons": [
            {condition.value: record.model_dump(mode="json") for condition, record in comparison.items()}
            for comparison in session.comparisons
        ],
        "single_turn_records": [record.model_dump(mode="json") for record in session.single_turn_records],
    }
    st.download_button(
        "Export Run Data (JSON)",
        data=json.dumps(payload, indent=2),
        file_name=f"{session.run_id}.json",
        mime="application/json",
        key="export_run_data",
    )


def _render_api_key_input(provider_name: str, widget_key_prefix: str, env_key_present: bool) -> str | None:
    """Renders this provider's credential controls and reports which
    credential is actually in effect: an environment variable, a key typed
    into this session, or none. Never displays, logs, exports, or folds any
    key into config_hash or a TurnRecord — type="password" masks the field
    on-screen, and the returned value is used only to construct the live
    client for this session.

    Each widget's own st.session_state entry is deleted by Streamlit the
    moment it isn't rendered on some rerun — which happens to every widget
    here while a different backend is selected. A second, plain (non-
    widget) "*_shadow" session_state entry survives that gap; each widget
    is re-seeded from its shadow the next time it's shown, so a typed key
    or a checked "use a different key" box outlives a backend switch away
    and back."""
    use_different_key = f"{widget_key_prefix}_use_different_key"
    use_different_shadow = f"{use_different_key}_shadow"
    api_key_key = f"{widget_key_prefix}_api_key_input"
    api_key_shadow = f"{api_key_key}_shadow"

    api_key_input = None
    if env_key_present:
        if use_different_key not in st.session_state:
            st.session_state[use_different_key] = st.session_state.get(use_different_shadow, False)
        use_different = st.checkbox("Use a different key", key=use_different_key)
        st.session_state[use_different_shadow] = use_different
        if use_different:
            if api_key_key not in st.session_state:
                st.session_state[api_key_key] = st.session_state.get(api_key_shadow, "")
            api_key_input = st.text_input(f"{provider_name} API key", type="password", key=api_key_key)
            st.session_state[api_key_shadow] = api_key_input
    else:
        if api_key_key not in st.session_state:
            st.session_state[api_key_key] = st.session_state.get(api_key_shadow, "")
        api_key_input = st.text_input(f"{provider_name} API key", type="password", key=api_key_key)
        st.session_state[api_key_shadow] = api_key_input
        if not api_key_input:
            st.warning(f"Enter an API key to use the {provider_name} backend, or switch to Offline.")

    if api_key_input:
        st.caption("Credential: Session key ✓")
    elif env_key_present:
        st.caption("Credential: Environment key detected ✓")
    else:
        st.caption("Credential: Not configured")

    return api_key_input


def main() -> None:
    st.set_page_config(page_title="Human–AI Dynamic Affective Cooperation", layout="wide")
    st.title("Human–AI Dynamic Affective Cooperation")
    st.caption(
        "A research prototype for studying how current and evolving affective information "
        "shapes AI support in human decision-making."
    )

    anthropic_env_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    openai_env_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    with st.sidebar:
        st.header("LLM backend")
        # Offline is always the default, regardless of which environment
        # keys are present: it is deterministic and guaranteed to
        # demonstrate the intended policy difference, while live behavior
        # can vary — a live backend should be a deliberate choice, not the
        # default a researcher lands on just because a key happens to be
        # set in their shell.
        default_index = BACKEND_OPTIONS.index(OFFLINE_BACKEND)
        backend = st.radio(
            "Backend", BACKEND_OPTIONS, index=default_index,
            captions=[
                "Prepared deterministic demo with no network calls.",
                "Live appraisal and response generation using OpenAI.",
                "Live appraisal and response generation using Anthropic.",
            ],
        )
        api_key_input = None
        if backend == OPENAI_BACKEND:
            api_key_input = _render_api_key_input("OpenAI", "openai", openai_env_key_present)
        elif backend == ANTHROPIC_BACKEND:
            api_key_input = _render_api_key_input("Anthropic", "anthropic", anthropic_env_key_present)

    if st.session_state.get("backend") != backend or "session" not in st.session_state:
        try:
            if backend == OFFLINE_BACKEND:
                session_obj, model_id = _init_mock_session()
            elif backend == OPENAI_BACKEND:
                if not (api_key_input or openai_env_key_present):
                    st.stop()
                session_obj, model_id = _init_openai_session(api_key_input)
            else:
                if not (api_key_input or anthropic_env_key_present):
                    st.stop()
                session_obj, model_id = _init_live_session(api_key_input)
            st.session_state.session = session_obj
            st.session_state.backend = backend
            st.session_state["active_model_id"] = model_id
        except ImportError as exc:
            st.error(str(exc))
            st.stop()

    session: SessionState = st.session_state.session

    with st.sidebar:
        st.caption(f"**{backend} · {st.session_state.get('active_model_id', '—')}**")
        _render_export_button(session)
        # Always available in the sidebar, next to backend/model status, so
        # it's reachable regardless of backend or whether any turn has run
        # yet.
        reset_requested = experiment_controls.render_reset_button()

    if reset_requested:
        # A brand new SessionState (rather than session.reset() on the
        # existing one) keeps this one reset path correct and uniform for
        # all three backends: a fresh run_id/history with no partial state
        # to reconcile, and a fresh live client is cheap to construct. It
        # also covers a backend switch for free — that already takes the
        # `st.session_state.get("backend") != backend` branch above on the
        # very next rerun, with no reset click needed.
        del st.session_state["backend"]
        del st.session_state["session"]
        st.session_state.pop("last_replay", None)
        st.session_state.pop("active_model_id", None)
        st.rerun()

    # Offline and live backends show different controls: Offline's own
    # "Run Prepared Demo" always compares ALL_CONDITIONS, so a condition
    # selector or a live-turn rho override would do nothing there. Live
    # backends show both, since they apply to the next live turn.
    if backend == OFFLINE_BACKEND:
        st.subheader("Prepared demo")
        st.caption(
            "Runs a fixed two-turn decision scenario across all three experimental conditions. "
            "No network connection is required."
        )
        if st.button("Run Prepared Demo", key="run_prepared_demo"):
            # Each press starts a fresh offline run rather than appending
            # onto whatever is already in this session: the fixture itself
            # has only two turns, so a second press without an explicit
            # Reset would otherwise re-run the same two turns under new
            # turn_ids (3, 4, ...) — same content, confusingly duplicated
            # in the Counterfactual Analysis turn selector.
            session_obj, model_id = _init_mock_session()
            st.session_state.session = session_obj
            st.session_state["active_model_id"] = model_id
            st.session_state.pop("last_replay", None)
            fixture = load_demo_fixture(DEMO_FIXTURE_PATH)
            for turn in fixture.turns:
                results = session_obj.controller.compare(
                    turn.user_text, turn.task_context, fixture.goal_state, ALL_CONDITIONS,
                )
                session_obj.record_comparison(results)
            st.rerun()
        display_conditions = ALL_CONDITIONS
    else:
        display_conditions = experiment_controls.render_condition_selector()
        rho_override = experiment_controls.render_temporal_persistence()
        st.subheader("Live Interaction")
        if not display_conditions:
            st.warning("Select at least one experimental condition.")
        with st.form("live_turn_form", clear_on_submit=True):
            user_text = st.text_area("Participant message")
            submitted = st.form_submit_button(
                "Run Turn", type="primary", key="run_turn", disabled=not display_conditions,
            )
        if submitted and user_text.strip() and display_conditions:
            results = session.controller.compare(
                user_text, {}, session.goal_state, display_conditions, rho_override=rho_override,
            )
            session.record_comparison(results)
            st.session_state.pop("last_replay", None)

    # Nothing below this point renders until at least one turn has been
    # recorded — no empty-state placeholders on the initial screen.
    latest = session.latest_comparison()
    if latest is None:
        return

    st.divider()

    # Counterfactual Analysis (DYNAMIC-only rho replay) renders nothing at
    # all until at least one DYNAMIC turn exists — see ui/
    # experiment_controls.py's own render_counterfactual_replay().
    replay_request = experiment_controls.render_counterfactual_replay(session)
    if replay_request is not None:
        turn_id, condition, replay_rho = replay_request
        try:
            replayed = session.controller.replay_turn(turn_id, condition, replay_rho)
        except KeyError as exc:
            st.error(str(exc))
        else:
            original = _find_original_record(session, turn_id, condition)
            st.session_state["last_replay"] = (original, replayed)

    last_replay = st.session_state.get("last_replay")
    if last_replay is not None:
        original, replayed = last_replay
        researcher_dashboard.render_replay_comparison(original, replayed)
        st.divider()

    interaction_view.render(latest, display_conditions)
    researcher_dashboard.render(latest, display_conditions)
    trajectory_view.render(session, display_conditions)


if __name__ == "__main__":
    main()
