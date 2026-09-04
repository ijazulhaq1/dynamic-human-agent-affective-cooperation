"""tests/test_observation_builder.py — M1 (ObservationBuilder), blueprint §6.1.

Phase 1 gate (blueprint §10 Phase 1), together with test_derived_state.py
and test_transition.py. Not itself named in the blueprint's §9 test map, but
required by the same phase: turn_id sequencing, timestamping, raw_history
population and reset are all M1's job and are exercised directly here so a
regression is caught before Pipeline (a later phase) starts relying on them.
"""

from datetime import datetime

from models.observation import Turn
from services.observation_builder import ObservationBuilder


def test_turn_id_sequencing_starts_at_one_and_increments():
    builder = ObservationBuilder()
    o1 = builder.build(user_text="first", task_context={}, raw_history=())
    o2 = builder.build(user_text="second", task_context={}, raw_history=())
    o3 = builder.build(user_text="third", task_context={}, raw_history=())
    assert (o1.turn_id, o2.turn_id, o3.turn_id) == (1, 2, 3)


def test_build_stamps_a_timestamp():
    builder = ObservationBuilder()
    o_t = builder.build(user_text="x", task_context={}, raw_history=())
    assert isinstance(o_t.timestamp, datetime)


def test_raw_history_is_the_callers_snapshot_not_empty():
    """The caller's pre-turn snapshot (Pipeline's history_before, a later
    phase) is what populates O_t.raw_history — never an empty placeholder,
    the bug the blueprint's own review round 2 fixed in ObservationBuilder."""
    builder = ObservationBuilder()
    prior_turn = Turn(turn_id=1, user_text="hello")
    snapshot = (prior_turn,)
    o_t = builder.build(user_text="follow-up", task_context={}, raw_history=snapshot)
    assert o_t.raw_history == [prior_turn]


def test_raw_history_defaults_to_empty_on_first_turn():
    builder = ObservationBuilder()
    o_t = builder.build(user_text="first ever turn", task_context={}, raw_history=())
    assert o_t.raw_history == []


def test_event_flags_extracted_from_task_context():
    builder = ObservationBuilder()
    o_t = builder.build(
        user_text="x",
        task_context={"event_flags": ["DEADLINE_MOVED", "OPTION_REMOVED"]},
        raw_history=(),
    )
    assert o_t.event_flags == ["DEADLINE_MOVED", "OPTION_REMOVED"]


def test_event_flags_default_to_empty_list_when_absent():
    builder = ObservationBuilder()
    o_t = builder.build(user_text="x", task_context={}, raw_history=())
    assert o_t.event_flags == []


def test_event_flags_default_to_empty_list_when_malformed():
    """task_context is caller-supplied scenario data, not a validated schema
    at this layer — a malformed event_flags value degrades to [] rather than
    raising, since a scenario-authoring typo here should not crash a turn."""
    builder = ObservationBuilder()
    o_t = builder.build(user_text="x", task_context={"event_flags": "not-a-list"}, raw_history=())
    assert o_t.event_flags == []


def test_reset_turn_counter_returns_sequencing_to_one():
    builder = ObservationBuilder()
    builder.build(user_text="a", task_context={}, raw_history=())
    builder.build(user_text="b", task_context={}, raw_history=())
    builder.reset_turn_counter()
    o_t = builder.build(user_text="c", task_context={}, raw_history=())
    assert o_t.turn_id == 1
