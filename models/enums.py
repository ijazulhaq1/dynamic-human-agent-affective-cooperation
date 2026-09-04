"""Canonical enums — §20.1 of the frozen specification, §4 of the Implementation Blueprint.

Every value here is copied verbatim from the frozen pre-implementation
specification. Do not add, rename, or reinterpret a member without a
specification revision (§19.11/§20.20) — this file is a restatement of a
frozen contract, not a place to make new design decisions.
"""

from enum import Enum


class Condition(str, Enum):
    TASK_FOCUSED = "TASK_FOCUSED"
    CURRENT_CUE = "CURRENT_CUE"
    DYNAMIC = "DYNAMIC"


class Policy(str, Enum):
    INFORM = "INFORM"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    CLARIFY = "CLARIFY"
    # CHALLENGE is never independently selected by any rule in policy_rules.yaml —
    # it is a realization mode of CLARIFY, kept only for logging/analysis
    # granularity (§20.1, final boundary §20.20).
    CHALLENGE = "CHALLENGE"
    REDIRECT = "REDIRECT"
    DEFER = "DEFER"


class EvidenceStrength(str, Enum):
    EXPLICIT = "EXPLICIT"
    STRONG_INDIRECT = "STRONG_INDIRECT"
    WEAK_INDIRECT = "WEAK_INDIRECT"
    CONTRADICTORY = "CONTRADICTORY"
    INSUFFICIENT = "INSUFFICIENT"


class GoalUpdateSource(str, Enum):
    HUMAN_EXPLICIT = "HUMAN_EXPLICIT"
    TASK_EVENT = "TASK_EVENT"
    RESEARCHER_CONFIG = "RESEARCHER_CONFIG"


class EstimatorStatus(str, Enum):
    OK = "OK"
    RETRY_OK = "RETRY_OK"
    FALLBACK_LOW_CONFIDENCE = "FALLBACK_LOW_CONFIDENCE"
    ERROR = "ERROR"


class GeneratorStatus(str, Enum):
    OK = "OK"
    FALLBACK_TEMPLATE = "FALLBACK_TEMPLATE"
    ERROR = "ERROR"


class RationaleCode(str, Enum):
    SAFETY_OVERRIDE = "SAFETY_OVERRIDE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_CONTROL = "LOW_CONTROL"
    REDIRECT_ELIGIBLE = "REDIRECT_ELIGIBLE"
    CLARIFY_NEEDED = "CLARIFY_NEEDED"
    INFORM_NEEDED = "INFORM_NEEDED"
    ACKNOWLEDGE_ELIGIBLE = "ACKNOWLEDGE_ELIGIBLE"
    MINIMAL_SUPPORT = "MINIMAL_SUPPORT"
