"""tests/test_app_smoke.py — app.py + ui/*.py (Phase 7). Not named in the
blueprint's own test map (the blueprint's §9 test-suite map stops at
test_experiment_controller.py; no ui/app test file is ever named there,
and Phase 7's own gate — "Full live rehearsal × 10 (§16.2, §20.17);
offline/replay fallback verified with network disabled" — is explicitly a
MANUAL researcher acceptance step, not a pytest target, per README.md's
Phase 7 section). Added anyway, per this project's established pattern of
testing every module it builds, using Streamlit's own headless
streamlit.testing.v1.AppTest — this is the automatable SLICE of that manual
gate: it launches the real app.py, in the real Mock (offline/replay)
backend, with NO network access and NO API key, and drives it through a
full demo-scenario run end to end, asserting no exception is raised at any
step. It does not and cannot replace the researcher's own live, 10-turn,
real-API-key rehearsal — see this module's own test names below for
exactly which slice each one covers.
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


def test_app_loads_in_mock_backend_by_default_with_no_api_key(monkeypatch):
    """No ANTHROPIC_API_KEY in the environment (this test's own sandbox —
    monkeypatch.delenv guards against it leaking in from a real one) means
    the sidebar backend radio must default to Mock (offline/replay), and
    the initial render must succeed with zero network calls — nothing in
    _init_mock_session() (app.py) ever imports or references the
    `anthropic` package at all."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _run_app()
    assert at.sidebar.radio[0].value == "Mock (offline/replay)"


def test_demo_fixture_button_runs_both_turns_with_no_exception(monkeypatch):
    """The offline/replay acceptance path itself: clicking "Run
    demo_fixture.yaml" drives both of config/demo_fixture.yaml's turns
    through all three conditions via the real ExperimentController.compare()
    — the same call path tests/test_demo_fixture.py verifies analytically —
    and the dashboard/trajectory views render the result without raising."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _run_app()
    demo_button = next(b for b in at.button if "demo_fixture" in b.label)
    demo_button.click().run()
    assert not at.exception, f"demo scenario run raised: {at.exception}"
    # Six tabs render: three (researcher_dashboard) + three (trajectory_view),
    # one per condition each — confirms the comparison actually populated
    # SessionState.comparisons, not just that no exception happened to occur.
    assert len(at.tabs) == 6


def test_new_run_reset_after_demo_scenario_does_not_raise(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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


def test_live_backend_without_api_key_stops_cleanly(monkeypatch):
    """Switching to the Anthropic (live) backend with no key available
    anywhere (env var absent, sidebar text_input left empty) must st.stop()
    — not crash, not silently construct a real network-calling client with
    no credential."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _run_app()
    at.sidebar.radio[0].set_value("Anthropic (live)").run()
    assert not at.exception, f"live backend with no key raised instead of stopping cleanly: {at.exception}"
