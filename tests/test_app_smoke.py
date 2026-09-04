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

Post-delivery redesign (user's own 15-item frontend-redesign spec, item 13
— see /root/.claude/plans/validated-cooking-wolf.md): every renamed button
now carries a stable `key=` (app.py/ui/experiment_controls.py), and every
button lookup below was switched from label-substring matching (fragile —
breaks the moment copy is edited) to `at.button(key=...)`/`at.get_by_key
(key=...)`, per the user's own explicit instruction: "Prefer stable widget
keys rather than making tests depend on legacy label substrings... Do not
preserve awkward UI wording merely because existing tests search for that
wording. Update the tests." Tab-count assertions also changed from 6 to 3:
ui/researcher_dashboard.py's render() no longer builds st.tabs() at all
(item 7 replaced its three per-condition tabs with one unified comparison
table) — only ui/trajectory_view.py's own three per-condition tabs remain.

Post-delivery redesign, round 2 (user's own detailed correctness/UX review
of the round-1 redesign — see README.md's "Frontend redesign, round 2"
section for the full account): Offline and live backends now show
genuinely different controls (no more "Conditions to compare"/live-only rho
override sitting uselessly in Offline mode), nothing renders before the
first turn (no more three redundant "No turns yet" captions), Counterfactual
Replay is DYNAMIC-only with no condition selector and stays hidden until a
DYNAMIC turn exists, and a typed API key now survives a full backend
switch away and back (previously it did not — see app.py's own
_render_api_key_input() docstring for exactly why, and the fix). The tests
at the bottom of this file (test_offline_backend_shows_..., test_live_
backend_shows_..., test_research_tools_..., test_typed_api_key_survives_...)
are new for this round; every test above this point already existed and
needed only the "Condition to replay" removal reflected."""

import json
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
    """The offline/replay acceptance path itself: clicking "Run Prepared
    Demo" drives both of config/demo_fixture.yaml's turns through all three
    conditions via the real ExperimentController.compare() — the same call
    path tests/test_demo_fixture.py verifies analytically — and the
    dashboard/trajectory views render the result without raising."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    at.button(key="run_prepared_demo").click().run()
    assert not at.exception, f"demo scenario run raised: {at.exception}"
    # Three tabs render (ui/trajectory_view.py, one per condition) —
    # researcher_dashboard.py no longer builds tabs of its own (item 7
    # replaced its three per-condition tabs with one unified table) — this
    # confirms the comparison actually populated SessionState.comparisons,
    # not just that no exception happened to occur.
    assert len(at.tabs) == 3


def test_new_run_reset_after_demo_scenario_does_not_raise(monkeypatch):
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    at.button(key="run_prepared_demo").click().run()
    assert not at.exception

    at.button(key="reset_session").click().run()
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
    at.button(key="run_prepared_demo").click().run()
    assert not at.exception

    at.button(key="reset_session").click().run()
    assert not at.exception

    at.button(key="run_prepared_demo").click().run()
    assert not at.exception, f"second demo run after reset raised: {at.exception}"
    assert len(at.tabs) == 3


def test_running_prepared_demo_twice_without_reset_does_not_duplicate_turns(monkeypatch):
    """The user's own finding, testing the delivered app directly: pressing
    "Run Prepared Demo" a second time (with no Reset Session click in
    between) used to APPEND config/demo_fixture.yaml's two turns onto the
    existing session rather than replacing it — producing turn_ids 1-4
    with turns 3/4 repeating turns 1/2's text verbatim, confusingly
    duplicated in the Counterfactual Analysis "Turn to replay" selector.
    Each press now starts a fresh offline SessionState first, so however
    many times "Run Prepared Demo" is pressed, exactly turn_ids 1 and 2
    ever exist — matching the fixture's own two turns."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    at.button(key="run_prepared_demo").click().run()
    assert not at.exception

    at.button(key="run_prepared_demo").click().run()
    assert not at.exception, f"second Run Prepared Demo press raised: {at.exception}"

    session = at.session_state["session"]
    assert len(session.comparisons) == 2
    turn_ids = sorted({record.turn_id for comparison in session.comparisons for record in comparison.values()})
    assert turn_ids == [1, 2]

    turn_select = next(sb for sb in at.selectbox if sb.label == "Turn to replay")
    assert len(turn_select.options) == 2
    assert turn_select.options[0].startswith("Turn 1 —")
    assert turn_select.options[1].startswith("Turn 2 —")


def test_counterfactual_analysis_shows_selected_turn_while_latest_interaction_stays_on_the_latest_turn(monkeypatch):
    """The user's own finding, testing the delivered app directly: picking
    Turn 1 in "Turn to replay" left the "Interaction" section (renamed here
    to "Latest Interaction") showing Turn 2's text regardless — the
    counterfactual calculation itself was already correct (verified
    separately against the real TurnRecord data), but nothing on screen
    said the two sections were showing two different turns. Fixed by
    surfacing the selected turn's own participant text directly under
    "Turn to replay" (render_counterfactual_replay(), ui/
    experiment_controls.py) as soon as the selection changes — no "Run
    Counterfactual" click required — and renaming the lower section
    "Interaction" -> "Latest Interaction" so its own scope is explicit.
    This drives exactly the scenario the user reported: select Turn 1,
    confirm Counterfactual Analysis shows Turn 1's own text, confirm Latest
    Interaction is unchanged and still shows Turn 2's."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    at.button(key="run_prepared_demo").click().run()
    assert not at.exception

    session = at.session_state["session"]
    dynamic_records_by_turn = {
        record.turn_id: record
        for comparison in session.comparisons
        for condition, record in comparison.items()
        if condition == Condition.DYNAMIC
    }
    turn1_text = dynamic_records_by_turn[1].observation.user_text
    turn2_text = dynamic_records_by_turn[2].observation.user_text
    assert turn1_text != turn2_text  # sanity: the fixture's two turns say different things

    turn_select = next(sb for sb in at.selectbox if sb.label == "Turn to replay")
    turn_select.set_value(1).run()
    assert not at.exception, f"selecting Turn 1 in the replay selector raised: {at.exception}"

    markdown_values = [m.value for m in at.markdown]
    assert any(f"**Participant:** {turn1_text}" in v for v in markdown_values), (
        "Counterfactual Analysis did not show Turn 1's own participant text after selecting Turn 1"
    )
    assert any(f"**Participant:** {turn2_text}" in v for v in markdown_values), (
        "Latest Interaction should still show Turn 2's text, unaffected by the replay selection"
    )

    subheaders = {s.value for s in at.subheader}
    assert "Latest Interaction" in subheaders
    assert "Interaction" not in subheaders


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
    at.button(key="run_prepared_demo").click().run()
    assert not at.exception, f"initial demo run raised: {at.exception}"

    # Redesign round 2 (user's own finding): Counterfactual Replay is now
    # DYNAMIC-only — no "Condition to replay" selector at all (see ui/
    # experiment_controls.py's own render_counterfactual_replay()
    # docstring) — so replaying turn 2 no longer needs to select a
    # condition, only a turn.
    turn_select = next(sb for sb in at.selectbox if sb.label == "Turn to replay")
    replay_button = at.button(key="run_counterfactual")

    turn_select.set_value(2)
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
    assert "OpenAI" in at.sidebar.caption[-1].value  # "**OpenAI · <model_id>**" (item 2's compact status caption)

    at.text_area[0].set_value("I'm torn on this decision.")
    at.button(key="run_turn").click().run()

    assert not at.exception, f"live turn through the OpenAI backend raised: {at.exception}"
    assert len(at.tabs) == 3  # trajectory_view, one tab per condition


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
    at.button(key="run_prepared_demo").click().run()
    assert not at.exception
    assert len(at.tabs) == 3  # Offline demo has run and populated comparisons

    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception, f"switching backend after a demo run raised: {at.exception}"
    assert len(at.tabs) == 0  # fresh SessionState — no leftover Offline-session tabs


# ---------------------------------------------------------------------------
# Redesign round 2 (user's own detailed correctness/UX review) — backend-
# aware controls, no empty-state clutter, DYNAMIC-only Counterfactual
# Replay hidden until eligible, and API-key persistence across a backend
# switch.
# ---------------------------------------------------------------------------


def test_offline_backend_shows_no_conditions_selector_or_temporal_persistence(monkeypatch):
    """The user's own biggest-named finding: Offline's "Run Prepared Demo"
    always compares ALL_CONDITIONS, so a "Conditions to compare" selector
    (which nothing ever reads in that path) and a live-turn-only rho
    override are both misleading in Offline mode — neither should render at
    all, before OR after the demo has run."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    assert len(at.multiselect) == 0
    assert not any(cb.label == "Override Dynamic ρ for next turn" for cb in at.checkbox)

    at.button(key="run_prepared_demo").click().run()
    assert not at.exception
    assert len(at.multiselect) == 0
    assert not any(cb.label == "Override Dynamic ρ for next turn" for cb in at.checkbox)


def test_live_backend_shows_conditions_selector_and_temporal_persistence(monkeypatch):
    """Mirror of the Offline check above: live backends genuinely use both
    controls (the next live turn's compare() call), so they belong there —
    and, per the redesign, are shown from the start, not only after a first
    turn. Uses a fake OPENAI_API_KEY (same safe construction as the
    existing OpenAI fake-adapter tests below — openai.OpenAI(api_key=...)
    makes no network call by itself) purely to get past the "no credential"
    st.stop() gate and reach the live-turn controls; no turn is submitted
    here, so the adapter's own extract/generate methods never need
    patching."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key-not-real")
    at = _run_app()
    # Offline is always the default backend regardless of which env keys are
    # present (round 3) — select OpenAI explicitly to reach the live-turn
    # controls this test is actually about.
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception
    assert len(at.multiselect) == 1
    assert any(cb.label == "Override Dynamic ρ for next turn" for cb in at.checkbox)


def test_no_result_sections_render_before_any_turn(monkeypatch):
    """The user's own finding: three redundant empty-state captions
    ("No turns yet...", "No turn recorded yet.", "No turns recorded yet.")
    used to render before anything had happened. None of Interaction/
    Researcher dashboard/Trajectory/Research Tools should render at all —
    not even their own empty-state message — until session.
    latest_comparison() is non-None."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    subheaders = {s.value for s in at.subheader}
    assert subheaders == {"Prepared demo"}
    assert len(at.tabs) == 0


def test_counterfactual_analysis_hidden_until_a_dynamic_turn_exists_then_dynamic_only(monkeypatch):
    """Item from the round-2 review: the counterfactual-replay tool must not
    appear (not even a "nothing to replay yet" message) before a DYNAMIC
    turn exists, and once it does appear, it must offer no "Condition to
    replay" selector at all — the tool is DYNAMIC-only by construction now
    (see ui/experiment_controls.py's own render_counterfactual_replay()
    docstring). Round 3 renamed this section "Counterfactual Analysis"."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    assert "Counterfactual Analysis" not in {s.value for s in at.subheader}

    at.button(key="run_prepared_demo").click().run()
    assert not at.exception
    assert "Counterfactual Analysis" in {s.value for s in at.subheader}
    selectbox_labels = [sb.label for sb in at.selectbox]
    assert selectbox_labels == ["Turn to replay"]
    assert "Original ρ" in [m.label for m in at.metric]


def test_typed_openai_api_key_survives_a_backend_switch_round_trip(monkeypatch):
    """The user's own explicit finding: "Typed API keys are not retained in
    st.session_state. That means if there is no environment key, switching
    backend or causing a rebuild can require re-entry." Verified directly
    against this Streamlit version that a keyed widget's own session_state
    entry is dropped the moment it isn't rendered for a run (exactly what
    happens while a different backend is selected) — app.py's own
    _render_api_key_input() now shadows each such value into a second,
    plain session_state entry that survives that gap and re-seeds the
    widget the next time it's shown. This test drives the exact round-trip
    the user's own report named: type a key, switch away, switch back."""
    _delenv_all_provider_keys(monkeypatch)
    at = _run_app()
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception
    at.text_input(key="openai_api_key_input").set_value("sk-my-typed-key").run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Anthropic").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception

    assert at.text_input(key="openai_api_key_input").value == "sk-my-typed-key"


def test_use_a_different_key_checkbox_and_typed_key_survive_backend_switch(monkeypatch):
    """Same finding as above, for the "environment key present but the
    researcher wants to override it" path: both the "Use a different key"
    checkbox AND whatever was typed underneath it must survive a backend
    switch away and back, not just the plain no-env-key text field."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-present")
    at = _run_app()
    assert at.sidebar.radio[0].value == "Offline"  # Offline is always the default (round 3)
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception

    at.checkbox(key="openai_use_different_key").set_value(True).run()
    assert not at.exception
    at.text_input(key="openai_api_key_input").set_value("sk-override-key").run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Offline").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception

    assert at.checkbox(key="openai_use_different_key").value is True
    assert at.text_input(key="openai_api_key_input").value == "sk-override-key"


def test_typed_api_key_never_appears_in_exported_run_data_or_technical_details(monkeypatch):
    """The user's own manual acceptance check (round 3, item 14) includes
    confirming no key text appears in the exported JSON or Technical
    Details; this test automates that portion. A typed OpenAI key is never
    passed to TurnRecord, config_hash, or logging (see app.py's own
    _init_openai_session()/_render_api_key_input() docstrings) — it is used
    only to construct the openai.OpenAI() client for this session. This
    drives a real live turn with a typed key and checks both the exact
    payload _render_export_button() would serialize and every rendered
    st.table (Technical Details included) for that key."""
    _delenv_all_provider_keys(monkeypatch)
    _patch_openai_adapter_methods(monkeypatch)

    at = _run_app()
    at.sidebar.radio[0].set_value("OpenAI").run()
    assert not at.exception

    typed_key = "sk-my-typed-key-should-never-leak"
    at.text_input(key="openai_api_key_input").set_value(typed_key).run()
    assert not at.exception

    at.text_area[0].set_value("I'm torn on this decision.")
    at.button(key="run_turn").click().run()
    assert not at.exception

    # The export button (and the just-recorded turn's Technical Details)
    # only reflect this turn on the NEXT rerun — see app.py's own render
    # order (the sidebar export button is built before the live-turn form
    # is processed on the submitting run).
    at.run()
    assert not at.exception

    session = at.session_state["session"]
    payload = {
        "run_id": session.run_id,
        "comparisons": [
            {condition.value: record.model_dump(mode="json") for condition, record in comparison.items()}
            for comparison in session.comparisons
        ],
        "single_turn_records": [record.model_dump(mode="json") for record in session.single_turn_records],
    }
    exported_json = json.dumps(payload)
    assert typed_key not in exported_json, "typed API key leaked into the exportable run data"

    for table in at.table:
        assert typed_key not in json.dumps(table.value, default=str), (
            "typed API key leaked into a rendered table (Technical Details or elsewhere)"
        )
