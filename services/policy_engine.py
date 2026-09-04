"""M6 — PolicyEngine. Blueprint §6.5, spec §20.7/§20.8.

One Rule dataclass drives both the affect-enabled table (this module,
build_affect_rules) and the TASK_FOCUSED table (services/policy_engine_
task_focused.py, build_task_focused_rules) — the eight/four rows ARE the
code, so there is no second, hand-written if/elif chain either table could
drift out of sync with. config/policy_rules.yaml documents the same rows for
readability; tests/test_policy.py checks it stays in sync with the Rule
lists built here.

Implementation-review note (judgment calls): the blueprint's table cells for
several rules are compressed English rather than exact formulas — "INFORM if
evidence else CLARIFY", "CLARIFY if unc/amb else INFORM", "ACKNOWLEDGE if
independently eligible/eligible", "INFORM if required", "task requires
evidence" / "task requires factual evidence". PolicyEngine.select()'s own
signature (h_t, c_t, g_t, d_t, a_t, a_star — no o_t/task_context) rules out
reading O_t.task_context directly, so these are resolved from G_t/D_t only:

  - "evidence" (R_SAFETY/TF_SAFETY secondary) = D_t.evidence_ambiguity below
    the clarify.d_amb_min gate — i.e. not highly ambiguous. Chosen over
    H_t.evidence_strength specifically because TF_SAFETY needs the same
    predicate and TASK_FOCUSED structurally cannot read H_t; D_t's
    task-derived half is the one signal both tables can legitimately use.
  - "task requires evidence" / "task requires factual evidence" (R_INFORM,
    R_ACK's "if required", TF_TASK_CLARIFY, TF_INFORM) = D_t.ambiguity_source
    == "required_evidence_formula", i.e. this turn's O_t actually carried a
    required_evidence list (see services/derived_features.py) rather than a
    d_amb_override or no ambiguity signal at all.
  - "unc/amb" (R_LOW_CTRL's primary branch) reuses the same clarify.h_unc_min
    / clarify.d_amb_min gates R_CLARIFY itself uses, rather than inventing a
    second uncertainty/ambiguity threshold.
  - "independently eligible" / "eligible" (R_LOW_CTRL, R_REDIRECT, R_CLARIFY,
    R_INFORM secondaries) = R_ACK's own condition evaluates True on the same
    Ctx — "would ACKNOWLEDGE have fired on its own merits here".

These are documented, defensible choices, not the frozen specification's own
wording — flagged here for the same adversarial review the rest of this
prototype has gotten.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from models.agent_state import AgentState, AgentTargetState
from models.derived_state import DerivedInteractionState
from models.enums import Condition, Policy, RationaleCode
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from models.policy import PolicyState


@dataclass(frozen=True)
class Ctx:
    """Everything a Rule's condition/primary/secondary may read. h_t, c_t and
    a_t are None for TASK_FOCUSED — structurally, not by convention: nothing
    in the TASK_FOCUSED rule table ever dereferences them, and PolicyEngine.
    select() always constructs TASK_FOCUSED's Ctx with all three set to None
    regardless of what its own h_t/c_t/a_t parameters were passed."""

    h_t: HumanAppraisal | None
    c_t: float | None
    g_t: GoalState
    d_t: DerivedInteractionState
    a_t: AgentState | None


@dataclass(frozen=True)
class Rule:
    id: str
    condition: Callable[[Ctx], bool]
    primary: Policy | Callable[[Ctx], Policy]
    secondary: Policy | None | Callable[[Ctx], Policy | None]
    rationale: RationaleCode
    reads_a_t: bool = False       # True only for R_REDIRECT and R_INFORM's a_info clause


@dataclass(frozen=True)
class PolicyConfig:
    """Flattened view of config/default.yaml's policy_thresholds + the one
    goal_state threshold PolicyEngine also needs (autonomy_high_threshold).
    Constructor-injected, never hardcoded — the same discipline as
    AppraisalEstimator's confidence_map and LinearPersistenceTransition's
    transition weights."""

    safety_risk_gate: float
    low_confidence_gate: float
    low_control_gate: float
    redirect_h_int_min: float
    redirect_a_intv_min: float
    redirect_a_info_min: float
    redirect_c_t_min: float
    redirect_safety_risk_max: float
    clarify_h_unc_min: float
    clarify_d_amb_min: float
    clarify_d_goal_min: float
    inform_a_info_min: float
    acknowledge_h_int_min: float
    acknowledge_h_rel_min: float
    acknowledge_c_t_min: float
    task_focused_safety_risk_gate: float
    task_focused_clarify_d_amb_min: float
    task_focused_clarify_d_goal_min: float
    autonomy_high_threshold: float

    @classmethod
    def from_mapping(cls, config: dict) -> "PolicyConfig":
        """config is the full parsed config/default.yaml (or an equivalent
        mapping) — this reads policy_thresholds.* and goal_state.*, never a
        second hardcoded copy of a number that belongs in that file."""
        pt = config["policy_thresholds"]
        return cls(
            safety_risk_gate=pt["safety_risk_gate"],
            low_confidence_gate=pt["low_confidence_gate"],
            low_control_gate=pt["low_control_gate"],
            redirect_h_int_min=pt["redirect"]["h_int_min"],
            redirect_a_intv_min=pt["redirect"]["a_intv_min"],
            redirect_a_info_min=pt["redirect"]["a_info_min"],
            redirect_c_t_min=pt["redirect"]["c_t_min"],
            redirect_safety_risk_max=pt["redirect"]["safety_risk_max"],
            clarify_h_unc_min=pt["clarify"]["h_unc_min"],
            clarify_d_amb_min=pt["clarify"]["d_amb_min"],
            clarify_d_goal_min=pt["clarify"]["d_goal_min"],
            inform_a_info_min=pt["inform"]["a_info_min"],
            acknowledge_h_int_min=pt["acknowledge"]["h_int_min"],
            acknowledge_h_rel_min=pt["acknowledge"]["h_rel_min"],
            acknowledge_c_t_min=pt["acknowledge"]["c_t_min"],
            task_focused_safety_risk_gate=pt["task_focused"]["safety_risk_gate"],
            task_focused_clarify_d_amb_min=pt["task_focused"]["clarify_d_amb_min"],
            task_focused_clarify_d_goal_min=pt["task_focused"]["clarify_d_goal_min"],
            autonomy_high_threshold=config["goal_state"]["autonomy_high_threshold"],
        )


# ---------- shared predicates (used by both rule tables) ----------


def evidence_available(ctx: Ctx, cfg: PolicyConfig) -> bool:
    """Reads D_t only (never H_t), so both R_SAFETY and TF_SAFETY can share
    it despite TASK_FOCUSED structurally excluding H_t.

    Requires BOTH that this turn actually carried a required_evidence
    structure (task_requires_evidence) AND that it is sufficiently resolved
    (evidence_ambiguity below the clarify.d_amb_min gate) — implementation-
    review fix. Checking evidence_ambiguity alone conflated "no evidence
    requirement was represented this turn" (evidence_ambiguity=0.0 via the
    task_rule/no-signal default, services/derived_features.py) with "evidence
    exists and is resolved". A D_t with no required_evidence at all is not
    evidence that a safety concern has been addressed; it is the absence of
    the question being asked."""
    return task_requires_evidence(ctx) and ctx.d_t.evidence_ambiguity < cfg.clarify_d_amb_min


def task_requires_evidence(ctx: Ctx) -> bool:
    """True exactly when this turn's D_t was derived from an actual
    required_evidence list (services/derived_features.py), not an override
    or the no-signal default."""
    return ctx.d_t.ambiguity_source == "required_evidence_formula"


def _ack_eligible(ctx: Ctx, cfg: PolicyConfig) -> bool:
    """R_ACK's own condition — reused wherever the table says an
    ACKNOWLEDGE secondary applies "if eligible"/"if independently
    eligible". Affect-enabled only; never called from TASK_FOCUSED rules."""
    return (
        ctx.h_t is not None
        and ctx.c_t is not None
        and ctx.h_t.affect_intensity >= cfg.acknowledge_h_int_min
        and ctx.h_t.goal_relevance >= cfg.acknowledge_h_rel_min
        and ctx.c_t >= cfg.acknowledge_c_t_min
    )


def _uncertain_or_ambiguous(ctx: Ctx, cfg: PolicyConfig) -> bool:
    """R_LOW_CTRL's "unc/amb" branch — the same gates R_CLARIFY uses."""
    assert ctx.h_t is not None
    return ctx.h_t.uncertainty >= cfg.clarify_h_unc_min or ctx.d_t.evidence_ambiguity >= cfg.clarify_d_amb_min


# ---------- affect-enabled rule table (CURRENT_CUE / DYNAMIC), §20.7 ----------


def build_affect_rules(cfg: PolicyConfig) -> list[Rule]:
    return [
        Rule(
            id="R_SAFETY",
            condition=lambda ctx: ctx.g_t.safety_risk >= cfg.safety_risk_gate,
            primary=Policy.DEFER,
            secondary=lambda ctx: Policy.INFORM if evidence_available(ctx, cfg) else Policy.CLARIFY,
            rationale=RationaleCode.SAFETY_OVERRIDE,
        ),
        Rule(
            id="R_LOW_CONF",
            condition=lambda ctx: ctx.c_t is not None and ctx.c_t < cfg.low_confidence_gate,
            primary=Policy.CLARIFY,
            secondary=Policy.INFORM,
            rationale=RationaleCode.LOW_CONFIDENCE,
        ),
        Rule(
            id="R_LOW_CTRL",
            condition=lambda ctx: ctx.h_t is not None and ctx.h_t.perceived_control < cfg.low_control_gate,
            primary=lambda ctx: Policy.CLARIFY if _uncertain_or_ambiguous(ctx, cfg) else Policy.INFORM,
            secondary=lambda ctx: Policy.ACKNOWLEDGE if _ack_eligible(ctx, cfg) else None,
            rationale=RationaleCode.LOW_CONTROL,
        ),
        Rule(
            id="R_REDIRECT",
            condition=lambda ctx: (
                ctx.h_t is not None
                and ctx.a_t is not None
                and ctx.c_t is not None
                and ctx.h_t.affect_intensity >= cfg.redirect_h_int_min
                and ctx.a_t.intervention_readiness >= cfg.redirect_a_intv_min
                and ctx.a_t.decision_information_priority >= cfg.redirect_a_info_min
                and ctx.c_t >= cfg.redirect_c_t_min
                and ctx.g_t.safety_risk < cfg.redirect_safety_risk_max
            ),
            primary=Policy.REDIRECT,
            secondary=lambda ctx: Policy.ACKNOWLEDGE if _ack_eligible(ctx, cfg) else Policy.INFORM,
            rationale=RationaleCode.REDIRECT_ELIGIBLE,
            reads_a_t=True,
        ),
        Rule(
            id="R_CLARIFY",
            condition=lambda ctx: (
                ctx.h_t is not None
                and (
                    ctx.h_t.uncertainty >= cfg.clarify_h_unc_min
                    or ctx.d_t.evidence_ambiguity >= cfg.clarify_d_amb_min
                    or ctx.d_t.goal_conflict >= cfg.clarify_d_goal_min
                )
            ),
            primary=Policy.CLARIFY,
            secondary=lambda ctx: Policy.ACKNOWLEDGE if _ack_eligible(ctx, cfg) else Policy.INFORM,
            rationale=RationaleCode.CLARIFY_NEEDED,
        ),
        Rule(
            id="R_INFORM",
            condition=lambda ctx: (
                ctx.a_t is not None and ctx.a_t.decision_information_priority >= cfg.inform_a_info_min
            )
            or task_requires_evidence(ctx),
            primary=Policy.INFORM,
            secondary=lambda ctx: Policy.ACKNOWLEDGE if _ack_eligible(ctx, cfg) else None,
            rationale=RationaleCode.INFORM_NEEDED,
            reads_a_t=True,
        ),
        Rule(
            id="R_ACK",
            condition=lambda ctx: _ack_eligible(ctx, cfg),
            primary=Policy.ACKNOWLEDGE,
            secondary=lambda ctx: Policy.INFORM if task_requires_evidence(ctx) else None,
            rationale=RationaleCode.ACKNOWLEDGE_ELIGIBLE,
        ),
        Rule(
            id="R_DEFAULT",
            condition=lambda ctx: True,
            primary=Policy.INFORM,
            secondary=None,
            rationale=RationaleCode.MINIMAL_SUPPORT,
        ),
    ]


def _hard_constraints(g_t: GoalState, cfg: PolicyConfig) -> list[str]:
    """§20.7 preamble / §20.8. NO_UNDUE_INFLUENCE is unconditional — a hard
    normative safeguard alongside safety and autonomy, always attached to
    every PolicyState regardless of scenario config (implementation-review
    fix: this previously branched on g_t.no_undue_influence, a plain bool a
    researcher could set False to make the safeguard silently disappear from
    hard_constraints; GoalState.no_undue_influence is now Literal[True] for
    the same reason, so the model and the policy layer tell the same story)."""
    constraints: list[str] = ["NO_UNDUE_INFLUENCE"]
    if g_t.autonomy_weight >= cfg.autonomy_high_threshold:
        constraints.append("AUTONOMY_HIGH")
    return constraints


@dataclass(frozen=True)
class _RuleOutcome:
    """Internal — the part of a rule match that isn't yet a full PolicyState
    (state_could/did_influence_policy are computed by select(), not _run())."""

    primary: Policy
    secondary: Policy | None
    triggered_rules: list[str]
    rationale_code: RationaleCode


class PolicyEngine:
    def __init__(self, config: PolicyConfig, affect_rules: list[Rule], tf_rules: list[Rule]):
        self._cfg, self._affect_rules, self._tf_rules = config, affect_rules, tf_rules

    def select(
        self,
        condition: Condition,
        h_t: HumanAppraisal | None,
        c_t: float | None,
        g_t: GoalState,
        d_t: DerivedInteractionState,
        a_t: AgentState | None,
        a_star: AgentTargetState | None,
    ) -> PolicyState:
        hard = _hard_constraints(g_t, self._cfg)

        if condition is Condition.TASK_FOCUSED:
            # TASK_FOCUSED's Ctx is built with h_t/c_t/a_t forced to None
            # regardless of what was passed in — the caller (Pipeline, a
            # later phase) may still have computed H_t/c_t for logging, but
            # PolicyEngineTaskFocused structurally cannot read them.
            outcome = self._run(self._tf_rules, Ctx(h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None))
            return PolicyState(
                primary=outcome.primary,
                secondary=outcome.secondary,
                triggered_rules=outcome.triggered_rules,
                hard_constraints=hard,
                rationale_code=outcome.rationale_code,
                state_could_influence_policy=False,
                state_did_influence_policy=False,
            )

        ctx = Ctx(h_t=h_t, c_t=c_t, g_t=g_t, d_t=d_t, a_t=a_t)
        outcome = self._run(self._affect_rules, ctx)
        could = self._reached_a_t_sensitive_rule(self._affect_rules, ctx)
        did = False
        if could and a_star is not None:
            ctx_counterfactual = replace(ctx, a_t=AgentState(**a_star.model_dump()))
            cf_outcome = self._run(self._affect_rules, ctx_counterfactual)
            did = (cf_outcome.primary, cf_outcome.secondary) != (outcome.primary, outcome.secondary)

        return PolicyState(
            primary=outcome.primary,
            secondary=outcome.secondary,
            triggered_rules=outcome.triggered_rules,
            hard_constraints=hard,
            rationale_code=outcome.rationale_code,
            state_could_influence_policy=could,
            state_did_influence_policy=did,
        )

    def _run(self, rules: list[Rule], ctx: Ctx) -> _RuleOutcome:
        for rule in rules:
            if rule.condition(ctx):
                primary = rule.primary(ctx) if callable(rule.primary) else rule.primary
                secondary = rule.secondary(ctx) if callable(rule.secondary) else rule.secondary
                return _RuleOutcome(
                    primary=primary,
                    secondary=secondary,
                    triggered_rules=[rule.id],
                    rationale_code=rule.rationale,
                )
        raise AssertionError("no rule matched — the *_DEFAULT rule's condition must always be True")

    def _reached_a_t_sensitive_rule(self, rules: list[Rule], ctx: Ctx) -> bool:
        """True once evaluation passes R_SAFETY/R_LOW_CONF/R_LOW_CTRL without
        firing — i.e. reaches R_REDIRECT, the first rule with reads_a_t=True
        (§20.7). Returns True there unconditionally (before even checking
        R_REDIRECT's own condition): from this point on, A_t/A*_t could in
        principle have changed which rule fires, which is exactly what
        state_could_influence_policy asks."""
        for rule in rules:
            if rule.reads_a_t:
                return True
            if rule.condition(ctx):
                return False
        return False
