# Reference evaluation

This report records one verified offline run on 2026-08-15. It is a checked-in reference, not a
substitute for rerunning the evaluation:

```powershell
.\scripts\reproduce.ps1
.\.venv\Scripts\python.exe -m exoswarm.cli ablation
```

The separate [Featherless live-model validation](live-validation-report.md) records three fresh
online investigations per target and the resulting structured-output, scientific-quality,
blindness, and lock/reveal audits.

## Scientific outcomes

| Target | Recovered period | Catalog-period error | Adaptive experiment | Locked disposition |
|---|---:|---:|---|---|
| `TARGET-X17` | 3.739466865 d | 0.000027135 d | centroid localization | PLANETARY INTERPRETATION SURVIVES IMPLEMENTED VETTING |
| `TARGET-X42` | 3.031910460 d | 0.000007112 d | P/2–P–2P harmonic test | PLANETARY INTERPRETATION WEAK |

Both runs passed every applicable deterministic grader: cached-source integrity, result SHA-256,
pre-lock trace commitment, Evidence Ledger linkage, catalog isolation before lock, reveal ordering,
mandatory diagnostics, period tolerance, turn budget, candidate numeric provenance, agent-prose
numeric provenance, and tool/ledger schemas. Cross-case branch and disposition diversity passed.

## Adaptive versus fixed checklist

| Target | Policy | Experiments | Selected optional test | Final disposition | Repeats |
|---|---|---:|---|---|---:|
| `TARGET-X17` | adaptive | 11 | centroid localization | SURVIVES IMPLEMENTED VETTING | 0 |
| `TARGET-X17` | fixed | 11 | centroid localization | SURVIVES IMPLEMENTED VETTING | 0 |
| `TARGET-X42` | adaptive | 11 | harmonic test | PLANETARY INTERPRETATION WEAK | 0 |
| `TARGET-X42` | fixed | 11 | centroid localization | PLANETARY INTERPRETATION PLAUSIBLE | 0 |

The adaptive policy did not execute fewer experiments in this two-case suite. Its demonstrated
advantage is trajectory quality on the negative control: it selected the discriminating harmonic
test, resolved the half-period alias, and avoided the fixed checklist's overly favorable
disposition. Equal results remain reported as equal.

The reference run used `--offline`. Its trace therefore records failed provider calls and explicit
deterministic fallbacks; these are agent-inference availability events, not failed scientific
tools. No token usage is reported when no model provider is called.

## Verified build checks

- Full test suite: 100 passed.
- Ruff lint and format checks: passed.
- Four cached TESS products: SHA-256 and byte sizes verified.
- Canonical Mission Control runs: 8 non-empty charts and 15 data tables per target in AppTest.

The values above are deterministic measurements from the cached observations. They are not
LLM-generated probabilities and do not constitute professional planet confirmation.
