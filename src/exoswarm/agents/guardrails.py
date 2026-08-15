"""Mechanical guardrails for model-authored, user-visible decision prose."""

from __future__ import annotations

import re
from typing import Any

from exoswarm.domain.ledger import EvidenceLedger
from exoswarm.domain.models import CriticDecision, SkepticDecision
from exoswarm.domain.numeric_provenance import NumericProvenanceGuard

_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:TIC\s*\d+|TOI[-\s]?\d+(?:\.\d+)?|TESS[-\s]?OBJECT[-\s]?OF[-\s]?INTEREST\s*\d+)\b",
    re.IGNORECASE,
)


def sanitize_skeptic_decision(
    decision: SkepticDecision, ledger: EvidenceLedger
) -> tuple[SkepticDecision, dict[str, Any]]:
    guard = NumericProvenanceGuard(ledger)
    fields = {
        "explanation": decision.explanation,
        "expected_discriminating_result": decision.expected_discriminating_result,
        "stop_if": decision.stop_if,
    }
    repaired: dict[str, str | None] = {}
    violations: dict[str, int] = {}
    identity_repairs: list[str] = []
    for name, text in fields.items():
        if text is None:
            repaired[name] = None
            continue
        safe, count, identity_changed = _sanitize_text(text, guard)
        repaired[name] = safe
        if count:
            violations[name] = count
        if identity_changed:
            identity_repairs.append(name)

    outcomes: dict[str, str] = {}
    for name, text in decision.predicted_outcomes.items():
        safe_name = _IDENTIFIER_PATTERN.sub("IDENTITY_WITHHELD", name)
        safe, count, identity_changed = _sanitize_text(text, guard)
        outcomes[safe_name] = safe
        if count:
            violations[f"predicted_outcomes.{safe_name}"] = count
        if identity_changed:
            identity_repairs.append(f"predicted_outcomes.{safe_name}")
        if safe_name != name:
            identity_repairs.append("predicted_outcomes.key")

    reason_code = decision.reason_code
    if _IDENTIFIER_PATTERN.search(reason_code):
        reason_code = "IDENTITY_WITHHELD"
        identity_repairs.append("reason_code")

    return (
        decision.model_copy(
            update={
                **repaired,
                "reason_code": reason_code,
                "predicted_outcomes": outcomes,
            }
        ),
        {
            "numeric_claims_repaired": violations,
            "identity_fields_repaired": identity_repairs,
            "changed": bool(violations or identity_repairs),
        },
    )


def sanitize_critic_decision(
    decision: CriticDecision, ledger: EvidenceLedger
) -> tuple[CriticDecision, dict[str, Any]]:
    guard = NumericProvenanceGuard(ledger)
    reason, count, reason_identity_changed = _sanitize_text(decision.reason, guard)
    reason_code = decision.reason_code
    code_identity_changed = False
    if _IDENTIFIER_PATTERN.search(reason_code):
        reason_code = "IDENTITY_WITHHELD"
        code_identity_changed = True
    return (
        decision.model_copy(update={"reason": reason, "reason_code": reason_code}),
        {
            "numeric_claims_repaired": {"reason": count} if count else {},
            "identity_fields_repaired": [
                name
                for name, changed in (
                    ("reason", reason_identity_changed),
                    ("reason_code", code_identity_changed),
                )
                if changed
            ],
            "changed": bool(count or reason_identity_changed or code_identity_changed),
        },
    )


def _sanitize_text(text: str, guard: NumericProvenanceGuard) -> tuple[str, int, bool]:
    report = guard.validate(text)
    repaired = guard.repair(text) if report.violations else text
    identity_safe = _IDENTIFIER_PATTERN.sub("[identity withheld until result lock]", repaired)
    return identity_safe, len(report.violations), identity_safe != repaired


__all__ = ["sanitize_critic_decision", "sanitize_skeptic_decision"]
