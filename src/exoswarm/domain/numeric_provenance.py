"""Mechanical guardrail for numerical claims in agent-authored UI prose."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from pydantic import Field

from .ledger import EvidenceItem, EvidenceLedger
from .models import FrozenDomainModel

_NUMBER_PATTERN = re.compile(
    r"(?<![\w-])"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s*(?P<unit>ppm|percent|%|hours?|hrs?|hr|h|days?|d|sigma|σ|pixels?|px))?"
    r"(?![\w-])",
    re.IGNORECASE,
)

_MASK_PATTERNS = (
    re.compile(r"\bTARGET-[A-Z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\b(?:EV|REQ|SK|CR|TRACE)-[A-Z0-9-]+\b", re.IGNORECASE),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[0-9a-f]{32,64}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+\-Z]+)?\b"),
    re.compile(r"\bv\d+(?:\.\d+){1,3}\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+){2,3}\b"),
    re.compile(r"\bP\s*/\s*\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*P\b", re.IGNORECASE),
)


class NumericClaim(FrozenDomainModel):
    raw: str
    value: float
    unit: str | None = None
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    supported_by: list[str] = Field(default_factory=list)


class NumericProvenanceReport(FrozenDomainModel):
    text: str
    claims: list[NumericClaim] = Field(default_factory=list)

    @property
    def violations(self) -> list[NumericClaim]:
        return [claim for claim in self.claims if not claim.supported_by]

    @property
    def valid(self) -> bool:
        return not self.violations


class NumericProvenanceViolation(ValueError):
    def __init__(self, report: NumericProvenanceReport) -> None:
        self.report = report
        claims = ", ".join(claim.raw for claim in report.violations)
        super().__init__(f"unsupported numerical claim(s): {claims}")


@dataclass(frozen=True, slots=True)
class _AllowedValue:
    value: float
    unit: str | None
    source_id: str


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    value = unit.casefold().strip()
    aliases = {
        "hour": "hour",
        "hours": "hour",
        "hr": "hour",
        "hrs": "hour",
        "h": "hour",
        "day": "day",
        "days": "day",
        "d": "day",
        "percent": "percent",
        "%": "percent",
        "ppm": "ppm",
        "sigma": "sigma",
        "σ": "sigma",
        "pixel": "pixel",
        "pixels": "pixel",
        "px": "pixel",
    }
    return aliases.get(value, value)


def _infer_unit(name: str) -> str | None:
    normalized = name.casefold()
    suffixes = {
        "_hours": "hour",
        "_hour": "hour",
        "_days": "day",
        "_day": "day",
        "_ppm": "ppm",
        "_percent": "percent",
        "_sigma": "sigma",
        "_pixels": "pixel",
        "_pixel": "pixel",
    }
    return next((unit for suffix, unit in suffixes.items() if normalized.endswith(suffix)), None)


def _canonical_value(value: float, unit: str | None) -> tuple[float, str | None]:
    unit = _normalize_unit(unit)
    if unit == "hour":
        return value / 24.0, "time_days"
    if unit == "day":
        return value, "time_days"
    return value, unit


def _rounding_tolerance(raw_number: str, value: float) -> float:
    try:
        decimal = Decimal(raw_number.lower())
    except InvalidOperation:
        return max(abs(value) * 1e-9, 1e-12)
    exponent = decimal.as_tuple().exponent
    if "e" in raw_number.casefold():
        return max(abs(value) * 5e-7, 1e-12)
    if exponent < 0:
        return float(Decimal("0.5") * (Decimal(10) ** exponent))
    return max(abs(value) * 1e-12, 1e-12)


def _mask_non_measurement_tokens(text: str) -> str:
    characters = list(text)
    for pattern in _MASK_PATTERNS:
        for match in pattern.finditer(text):
            characters[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(characters)


class NumericProvenanceGuard:
    """Reject or repair measurements that have no deterministic provenance."""

    def __init__(self, ledger: EvidenceLedger | Iterable[EvidenceItem]) -> None:
        self._items = ledger.items if isinstance(ledger, EvidenceLedger) else tuple(ledger)

    def validate(
        self,
        text: str,
        *,
        evidence_ids: Iterable[str] | None = None,
        additional_values: Mapping[str, tuple[float, str | None]] | None = None,
    ) -> NumericProvenanceReport:
        selected = set(evidence_ids) if evidence_ids is not None else None
        allowed = self._allowed_values(selected, additional_values or {})
        masked = _mask_non_measurement_tokens(text)
        claims: list[NumericClaim] = []
        for match in _NUMBER_PATTERN.finditer(masked):
            raw_number = match.group("value")
            value = float(raw_number)
            unit = _normalize_unit(match.group("unit"))
            sources = self._find_sources(value, unit, raw_number, allowed)
            claims.append(
                NumericClaim(
                    raw=text[match.start() : match.end()],
                    value=value,
                    unit=unit,
                    start=match.start(),
                    end=match.end(),
                    supported_by=sources,
                )
            )
        return NumericProvenanceReport(text=text, claims=claims)

    def enforce(self, text: str, **kwargs: object) -> str:
        report = self.validate(text, **kwargs)
        if not report.valid:
            raise NumericProvenanceViolation(report)
        return text

    def repair(
        self,
        text: str,
        *,
        replacement: str = "[measurement unavailable]",
        **kwargs: object,
    ) -> str:
        report = self.validate(text, **kwargs)
        repaired = text
        for violation in sorted(report.violations, key=lambda item: item.start, reverse=True):
            repaired = repaired[: violation.start] + replacement + repaired[violation.end :]
        return repaired

    def _allowed_values(
        self,
        selected_ids: set[str] | None,
        additional: Mapping[str, tuple[float, str | None]],
    ) -> list[_AllowedValue]:
        values: list[_AllowedValue] = []
        for item in self._items:
            if selected_ids is not None and item.id not in selected_ids:
                continue
            for name, value in item.numerical_results.items():
                unit = item.result_units.get(name) or _infer_unit(name)
                values.append(_AllowedValue(float(value), unit, item.id))
            for name, uncertainty in item.uncertainties.items():
                unit = uncertainty.unit or item.result_units.get(name) or _infer_unit(name)
                values.append(
                    _AllowedValue(float(uncertainty.value), unit, f"{item.id}:uncertainty:{name}")
                )
            # Experiment settings (period limits, duration grids, clipping
            # thresholds, seeds, and similar controls) are provenance, but they
            # are not measured scientific results.  Admitting them here would
            # let model prose relabel a search-grid value as a measurement.
        for source_id, (value, unit) in additional.items():
            values.append(_AllowedValue(float(value), _normalize_unit(unit), source_id))
        return values

    @staticmethod
    def _find_sources(
        claimed_value: float,
        claimed_unit: str | None,
        raw_number: str,
        allowed: Iterable[_AllowedValue],
    ) -> list[str]:
        claim_canonical, claim_dimension = _canonical_value(claimed_value, claimed_unit)
        raw_tolerance = _rounding_tolerance(raw_number, claimed_value)
        if claimed_unit in {"hour", "day"}:
            raw_tolerance = abs(_canonical_value(raw_tolerance, claimed_unit)[0])
        sources: list[str] = []
        for entry in allowed:
            allowed_canonical, allowed_dimension = _canonical_value(entry.value, entry.unit)
            if claimed_unit is not None and allowed_dimension != claim_dimension:
                continue
            # Unitless prose may cite a value that has a recorded unit; an explicit
            # incompatible unit may not.
            tolerance = max(raw_tolerance, abs(allowed_canonical) * 1e-9, 1e-12)
            if math.isclose(
                claim_canonical,
                allowed_canonical,
                rel_tol=0,
                abs_tol=tolerance,
            ):
                sources.append(entry.source_id)
        return sorted(set(sources))
