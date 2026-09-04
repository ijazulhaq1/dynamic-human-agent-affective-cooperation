"""tests/test_app_smoke.py — app.py + ui/*.py (Phase 7). Not named in the
blueprint's own test map (the blueprint's §9 test-suite map stops at
test_experiment_controller.py; no ui/app test file is ever named there,
and Phase 7's own gate — "Full live rehearsal × 10 (§16.2, §20.17);
offline/replay fallback verified with network disabled" — is explicitly a
MANUAL researcher acceptance step, not a pytest target, per README.md's
Phase 7 section). Added anyway, per this project's established pattern of
testing every module it builds, using Streamlit's own headless
streamlit.testing.v1.AppTest — this is the automatable SLICE of that manual
gate: it launches the real app.py, in the real Offline backend, with NO
network access and NO API key, and drives it through a full demo-scenario
run end to end, asserting no exception is raised at any step. It does not
and cannot replace the researcher's own live, 10-turn, real-API-key
rehearsal — see this module's own test names below for exactly which slice
each one covers.

Backend labels updated for OpenAI support (added post-delivery, user
request): the sidebar radio is now a three-way "Offline" / "OpenAI" /
"Anthropic" selector (app.py's BACKEND_OPTIONS) — Phase 7's own original
two-way "Mock (offline/replay)" / "Anthropic (live)" labels are gone, and
every assertion below was updated to match. See the OpenAI-specific tests
near the bottom of this file for that addition's own coverage.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from models.enums import Condition

# AppTest.from_file() resolves a relative path against the file that CALLS
# it (this test file's own directory, tests/), not the process's cwd — so
# this must be absolute to reach the real app.py at the repo root.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"app.py raised on initial load: {at.exception}"
    return at


def _delenv_all_provider_keys(monkeypatch) -> None:
    """Guards every test below against either provider's real key leaking
    in from the actual environment this suite happens to run in — used
    anywhere a test's own point is what happens with NO credential
    available at all."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_app_loads_in_offline_backend_by_default_with_no_api_keys(monkeypatch):
    """No ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment means the
    sidebar backend radio must default to Offline, and the initial render
    must succeed with zero network calls — nothing in _init_mock_session()
    (app.py) ever imports or references the `anthropic` or `openai`
    packages at all."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    assert at.sidebar.radio[0].value == "Offline"


def test_demo_fixture_button_runs_both_turns_with_no_exception(monkeypatch):
    """The offline/replay acceptance path itself: clicking "Run
    demo_fixture.yaml" drives both of config/demo_fixture.yaml's turns
    through all three conditions via the real ExperimentController.compare()
    — the same call path tests/test_demo_fixture.py verifies analytically —
    and the dashboard/trajectory views render the result without raising."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    demo_button = next(b for b in at.button if "demo_fixture" in b.label)
    demo_button.click().run()
    assert not at.exception, f"demo scenario run raised: {at.exception}"
    # Six tabs render: three (researcher_dashboard) + three (trajectory_view),
    # one per condition each — confirms the comparison actually populated
    # SessionState.comparisons, not just that no exception happened to occur.
    assert len(at.tabs) == 6


def test_new_run_reset_after_demo_scenario_does_not_raise(monkeypatch):
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    next(b for b in at.button if "demo_fixture" in b.label).click().run()
    assert not at.exception

    next(b for b in at.button if "New Run" in b.label).click().run()
    assert not at.exception, f"reset raised: {at.exception}"


def test_demo_scenario_runs_again_cleanly_after_reset(monkeypatch):
    """Regression check for a real bug caught before delivery: an earlier
    draft of app.py's reset handler called SessionState.reset() on the
    EXISTING session, which starts a fresh run_id but leaves Mock mode's
    one-shot, pre-scripted extract/generate queues (see
    _init_mock_session()'s own docstring in app.py) already exhausted from
    the first demo run — so a second "Run demo_fixture.yaml" click after
    Reset would hit MockLLMAdapter's own "called more times than it was
    scripted for" AssertionError. Fixed by discarding and rebuilding the
    whole SessionState (a fresh adapter with a freshly reloaded queue) on
    reset instead of resetting the existing one in place — this test drives
    exactly that reload-and-run-again sequence."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    next(b for b in at.button if "demo_fixture" in b.label).click().run()
    assert not at.exception

    next(b for b in at.button if "New Run" in b.label).click().run()
    assert not at.exception

    next(b for b in at.button if "demo_fixture" in b.label).click().run()
    assert not at.exception, f"second demo run after reset raised: {at.exception}"
    assert len(at.tabs) == 6


def test_replay_after_full_demo_does_not_raise(monkeypatch):
    """User's own explicitly requested regression test (Phase 7 review,
    required fix #2): "Add a regression test: run demo -> replay Turn 2
    DYNAMIC -> no exception." This is the automated check behind the
    OfflineDemoAdapter fix (llm/adapter.py), which replaces
    MockLLMAdapter's finite, scriptable generate_responses queue in
    _init_mock_session() (app.py). Before that fix, running the full demo
    (which itself calls the generator once per turn per condition, 2 turns
    x 3 conditions = 6 generate() calls) would already exhaust a
    one-shot queue sized for exactly one demo pass — so Pipeline.
    replay_turn()'s own unconditional re-invocation of the generator
    (services/pipeline.py, frozen Phase 5 behavior, not changeable here)
    on the very next replay would hit MockLLMAdapter's own "called more
    times than it was scripted for" AssertionError. OfflineDemoAdapter has
    no such limit."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    next(b for b in at.button if "demo_fixture" in b.label).click().run()
    assert not at.exception, f"initial demo run raised: {at.exception}"

    turn_select = next(sb for sb in at.selectbox if sb.label == "Turn to replay")
    condition_select = next(sb for sb in at.selectbox if sb.label == "Condition to replay")
    replay_button = next(b for b in at.button if b.label == "Replay")

    turn_select.set_value(2)
    condition_select.set_value(Condition.DYNAMIC)
    replay_button.click()
    at.run()

    assert not at.exception, f"replay of turn 2 / DYNAMIC after a full demo run raised: {at.exception}"


def test_anthropic_backend_without_api_key_stops_cleanly(monkeypatch):
    """Switching to the Anthropic backend with no key available anywhere
    (env var absent, sidebar text_input left empty) must st.stop() — not
    crash, not silently construct a real network-calling client with no
    credential."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    at.sidebar.radio[0].set_value("Anthropic").run()
    assert not at.exception, f"Anthropic backend with no key raised instead of stopping cleanly: {at.exception}"


# ---------------------------------------------------------------------------
# OpenAI backend (added post-delivery, user request) — strictly alongside
# Offline and Anthropic above, never replacing either.
# ---------------------------------------------------------------------------


def test_openai_backend_without_api_key_stops_cleanly(monkeypatch):
    """Mirrors test_anthropic_backend_without_api_key_stops_cleanly above,
    for the new OpenAI backend: no key available anywhere must st.stop()
    cleanly, never construct a real network-calling client with no
    credential."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception, f"OpenAI backend with no key raised instead of stopping cleanly: {at.exception}"


def _fake_openai_extract_appraisal(self, o_t, g_t, repair=False):
    return {
        "goal_relevance": 0.5, "goal_congruence": 0.0, "uncertainty": 0.3,
        "perceived_control": 0.5, "agency": 0.5, "affect_intensity": 0.2,
        "possible_affect": None, "evidence_tags": ["fake_openai_smoke_test"],
        "evidence_strength": "WEAK_INDIRECT",
    }


def _fake_openai_generate(self, contract):
    return "[fake OpenAI response for testing — no network call made]"


def _patch_openai_adapter_methods(monkeypatch) -> None:
    """Fakes OpenAILLMAdapter's own two LLM-boundary methods (llm/
    adapter.py) rather than app.py's construction of it — this still
    exercises the REAL app.py wiring end to end (_init_openai_session(),
    _build_pipeline(), ExperimentController, SessionState, every ui/*.py
    renderer) with a REAL openai.OpenAI() client object underneath it
    (safe to construct — see tests/test_openai_adapter.py's own
    construction test — it makes no network call by itself); only the two
    calls that would otherwise hit the real network are faked, matching
    this test suite's own established fake-at-the-LLM-boundary pattern."""
    import llm.adapter as adapter_module

    monkeypatch.setattr(adapter_module.OpenAILLMAdapter, "extract_appraisal", _fake_openai_extract_appraisal)
    monkeypatch.setattr(adapter_module.OpenAILLMAdapter, "generate", _fake_openai_generate)


def test_openai_backend_initializes_and_runs_a_live_turn_with_fake_adapter(monkeypatch):
    """OpenAI backend, instantiated with a fake client/config path
    (explicit test requirement): a fake OPENAI_API_KEY env var lets
    _init_openai_session() construct a real openai.OpenAI() client and a
    real OpenAILLMAdapter/Pipeline/ExperimentController/SessionState, with
    only the two LLM-boundary methods faked (see
    _patch_openai_adapter_methods above) — then drives one live turn
    through it via the same "Live turn" form the Anthropic backend already
    uses, asserting no exception anywhere in that wiring."""
    _delenv_all_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key-not-real")
    _patch_openai_adapter_methods(monkeypatch)

    at = _run_app()
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception, f"selecting OpenAI backend raised: {at.exception}"
    assert "OpenAI" in at.sidebar.caption[-1].value  # "Active backend: **OpenAI** — model: ..."

    at.text_area[0].set_value("I'm torn on this decision.")
    next(b for b in at.button if "Send" in b.label).click().run()

    assert not at.exception, f"live turn through the OpenAI backend raised: {at.exception}"
    assert len(at.tabs) == 6  # researcher_dashboard (3) + trajectory_view (3), one per condition each


def test_backend_switch_does_not_contaminate_state(monkeypatch):
    """Requirement: "backend switching does not contaminate state." Runs
    the Offline demo scenario to completion (populating SessionState.
    comparisons), then switches to OpenAI (faked) — the newly-built
    SessionState must start with an EMPTY comparison history, not the
    Offline session's leftover turns; both researcher_dashboard.render()
    and interaction_view.render() only ever create tabs/content once
    latest_comparison() is non-None (see their own "No turns yet"/"No turn
    recorded yet" early-return branches), so zero tabs after the switch is
    direct evidence of a genuinely fresh SessionState, not a stale one."""
    _delenv_all_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key-not-real")
    _patch_openai_adapter_methods(monkeypatch)

    at = _run_app()
    # OPENAI_API_KEY is present, so the sidebar would default straight to
    # the OpenAI backend (this module's own default-selection rule in
    # app.py) — select Offline explicitly first so this test actually
    # exercises a genuine Offline -> OpenAI switch.
    at.sidebar.radio[0].set_value("Offline").run()
    next(b for b in at.button if "demo_fixture" in b.label).click().run()
    assert not at.exception
    assert len(at.tabs) == 6  # Offline demo has run and populated comparisons

    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception, f"switching backend after a demo run raised: {at.exception}"
    assert len(at.tabs) == 0  # fresh SessionState — no leftover Offline-session tabs
