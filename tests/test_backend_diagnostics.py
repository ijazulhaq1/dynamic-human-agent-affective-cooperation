"""Phase 7 live-provider diagnostics are persisted without changing science."""

from llm.adapter import GeneratorError
from models.enums import Condition, EstimatorStatus, GeneratorStatus
from tests.pipeline_fixtures import (
    DEFAULT_OUTCOME_BASELINE_PRIORITIES,
    build_pipeline,
    default_goal_state,
)


class DiagnosticFailingAdapter:
    """Minimal provider double: both appraisal attempts and generation fail safely."""

    def __init__(self):
        self.last_error = None

    def extract_appraisal(self, o_t, g_t, repair=False):
        self.last_error = (
            "AuthenticationError: Incorrect API key provided: sk-[REDACTED]"
        )
        return None

    def generate(self, contract):
        self.last_error = "RateLimitError: current quota exceeded"
        raise GeneratorError("current quota exceeded")


def test_pipeline_persists_estimator_and_generator_diagnostics(tmp_path):
    adapter = DiagnosticFailingAdapter()
    pipeline, _, _ = build_pipeline(tmp_path, adapter=adapter, model_id="gpt-test")
    pipeline.outcome_baseline_store.elicit("run-diagnostic", DEFAULT_OUTCOME_BASELINE_PRIORITIES)

    record = pipeline.run_turn(
        run_id="run-diagnostic",
        user_text="I am unsure.",
        task_context={},
        condition=Condition.CURRENT_CUE,
        rho=0.0,
        g_t=default_goal_state(),
    )

    assert record.estimator_status is EstimatorStatus.FALLBACK_LOW_CONFIDENCE
    assert record.generator_status is GeneratorStatus.FALLBACK_TEMPLATE
    assert record.error_messages == [
        "Estimator backend: AuthenticationError: Incorrect API key provided: sk-[REDACTED]",
        "Generator backend: RateLimitError: current quota exceeded",
    ]
    assert all("secret" not in m.lower() for m in record.error_messages)
