# ExoSwarm

**AI mission control for agent-orchestrated exoplanet investigation.**

ExoSwarm investigates real cached NASA TESS observations, measures transit-like signals with
deterministic astronomy software, asks a bounded Skeptic which falsification experiment is most
useful next, has a separate Critic review that choice, and locks the result before external catalog
truth becomes callable.

> Agents decide which permitted scientific operation should happen next. Deterministic scientific
> software performs the operation and produces every measurement.

ExoSwarm does **not** claim to discover or confirm planets. Its strongest disposition is that a
planetary interpretation survived the limited implemented photometric/pixel vetting. Professional
confirmation remains an external catalog fact and may require observations this project does not
have.

## Quick start

Python 3.12 or newer is required. The repository includes four cached SPOC products (two light
curves and two target-pixel files, about 99 MiB total) through Git LFS.

```powershell
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

# Verify every cached NASA product before doing science.
.\.venv\Scripts\python.exe -m exoswarm.cli verify-cache

# See only metadata that is safe before reveal.
.\.venv\Scripts\python.exe -m exoswarm.cli targets

# Use a fresh root because locked investigations are deliberately write-once.
$quickRuns = "runs\quickstart-001" # Change the suffix when repeating the walkthrough.

# Run, inspect the lock, then reveal in a separate command.
.\.venv\Scripts\python.exe -m exoswarm.cli run TARGET-X17 --offline --runs-root $quickRuns
.\.venv\Scripts\python.exe -m exoswarm.cli reveal TARGET-X17 --runs-root $quickRuns

# The negative control follows a different evidence-driven branch.
.\.venv\Scripts\python.exe -m exoswarm.cli run TARGET-X42 --offline --runs-root $quickRuns

# Mission control reads the real run/ledger artifacts.
.\.venv\Scripts\python.exe -m exoswarm.cli ui --runs-root $quickRuns
```

The `--offline` mode is a safe, deterministic decision fallback and is labeled as such in the
trace. For the actual model-directed demo, copy `.env.example`, set `EXOSWARM_API_KEY`, and omit
`--offline`. The intended provider is Featherless's OpenAI-compatible chat-completions endpoint;
all provider configuration is isolated from the science/runtime packages.

For a fail-closed live validation run, use a fresh root and add `--require-live`. The command exits
nonzero if a Skeptic/Critic decision or executed adaptive selection used deterministic fallback:

```powershell
$liveRuns = "runs\live-demo-001"
.\.venv\Scripts\python.exe -m exoswarm.cli run TARGET-X17 --runs-root $liveRuns --require-live
.\.venv\Scripts\python.exe -m exoswarm.cli run TARGET-X42 --runs-root $liveRuns --require-live
```

Every decision is trace-classified as `LIVE_MODEL`, `REPAIRED_LIVE_MODEL`, or
`DETERMINISTIC_FALLBACK`. Mission Control displays that source together with provider, model,
structured attempts, and provider request IDs. It never displays the API key or raw model output.

`requirements.lock` records the exact Windows/Python 3.14 environment used for the verified
build. The version ranges in `pyproject.toml` remain the portable installation contract for
Python 3.12 and newer.

## Reproduce the claims

On systems with `make`:

```text
make reproduce
```

On Windows PowerShell:

```powershell
.\scripts\reproduce.ps1
```

Both commands verify cached source hashes, run the full test suite, rerun both deterministic
investigations without astronomy-network access, verify lock/reveal ordering, and write a new
timestamped evaluation report. Existing locked runs are never overwritten.

Useful focused commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m exoswarm.cli verify TARGET-X17
.\.venv\Scripts\python.exe -m exoswarm.cli ablation
```

The ablation actually runs the declared fixed policy
`BLS → odd/even → secondary → centroid → result` against the adaptive policy. It reports measured
experiment counts, branches, repeated calls, schema failures, fallbacks, disposition, latency, and
token usage where a provider reports it. Equal measurements remain equal; the report does not
manufacture an agent advantage.

## Curated blinded cases

The pre-reveal interface exposes only `TARGET-X17` and `TARGET-X42`. Their cached science manifests
contain opaque IDs, source hashes, cadence/sector information, and scientifically necessary pixel
geometry. Recognizable identities and catalog truth live in a separate backend-private manifest.

- `TARGET-X17` has a strong periodic transit-like signal and a real nearby-source concern. Its
  clean odd/even and secondary checks leave contamination as the useful challenge, so the Skeptic
  selects centroid localization.
- `TARGET-X42` has alternating deep/shallow events and a near-degenerate half-period cadence. The
  Skeptic instead selects the P/2–P–2P harmonic experiment, resolving the orbital period and
  strengthening the eclipsing-binary interpretation.

The repository deliberately does not publish the opaque-ID mapping in pre-reveal documentation.
After a result is locked, `reveal.json` and Mission Control show the actual target identity,
external catalog status, and catalog measurements. Those catalog facts remain separate from
ExoSwarm's disposition. For offline reproducibility, the reveal uses a pinned backend-private
catalog snapshot with source URLs; it is not a live catalog query during the demo.

The blindness threat model is an application capability boundary: agent and science packages
cannot call or serialize the private catalog before lock. A repository maintainer could still
inspect the versioned backend-private manifest or identity-bearing FITS headers. For a blind study
involving human operators, host that private data behind a separately administered service.

## Scientific path

The code-enforced baseline is:

```text
cached SPOC LC + TPF
→ QUALITY==0 inspection
→ normalization and transit-preserving bright-tail cleaning
→ limited gap-aware running-median detrending
→ Astropy Box Least Squares + phase fold
→ period / epoch / depth / duration / SNR / event count with uncertainty or tolerance
→ signal-quality gate
→ odd/even test
→ phase-0.5 secondary test
→ basic contamination screen
```

Adaptive tools then include the P/2–P–2P harmonic comparison, genuine target-pixel centroid/difference
imaging, and an allowed alternate-detrending sensitivity check. Every function returns a strict
`ScientificResult` or a structured failure; no tool returns free-form scientific prose.

The declared hypothesis set is planetary transit, eclipsing binary, neighboring/background
contaminant, stellar variability, instrumental/systematic artifact, and period alias/harmonic.
Evidence updates use a documented heuristic table. The weights are directional, uncalibrated model
assumptions—not planet probabilities—and the UI never labels them as confidence percentages.

## Auditability and blindness

Each run contains:

```text
runs/TARGET-X17/
  artifacts/artifacts.json
  artifacts/science/*.npz
  artifacts/science/*.png
  evidence.jsonl
  trace.jsonl
  result.json
  result.json.sha256
  reveal.json                 # exists only after a verified lock
```

The Evidence Ledger and trace are append-only and hash chained. `result.json` has no identity or
catalog field. The ground-truth gate verifies that result and its SHA-256 commit marker before it
can reveal anything. Automated tests cover the exact invariants
`test_catalog_unreachable_before_lock` and `test_target_identity_not_exposed`, plus a static import
audit that prevents the agent/science packages from importing the catalog capability.

Model responses are schema validated, repaired at most once, then replaced by an explicit safe
fallback. Agent-authored UI prose passes a numeric-provenance guard; unsupported values and guessed
TIC/TOI identities are mechanically withheld rather than merely discouraged in a prompt.

## Documentation

- [Specification extraction and scope](docs/specification-extract.md)
- [Architecture, state machine, and run contract](docs/architecture.md)
- [Scientific methods and limitations](docs/science-methods.md)
- [Reference evaluation and adaptive-vs-fixed result](evals/reference-report.md)
- [Six-run Featherless live-model validation](evals/live-validation-report.md)
- [Demo flow](docs/demo.md)

Primary data/catalog references:

- [MAST TESS data products](https://archive.stsci.edu/missions-and-data/tess/data-products)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [MAST TESS Eclipsing Binaries](https://archive.stsci.edu/hlsp/tess-ebs)
- [Featherless API quickstart](https://featherless.ai/docs/quickstart-guide)

## Repository map

```text
src/exoswarm/
  agents/       bounded provider, Skeptic, Critic, structured-output and prose guardrails
  domain/       schemas, Evidence Ledger, trace, registry, hypotheses, provenance guard
  science/      deterministic TESS/BLS/vetting/pixel computations
  security/     opaque vault, import boundary, result lock and catalog gate
  runtime/      Scientific Director and private target loader
  evaluation/   constraint graders and fixed-policy ablation
  ui/           Streamlit mission control over validated artifacts
data/           cached opaque science products and separate private catalog manifest
evals/          checked-in reference evaluation and adaptive-vs-fixed outcome
tests/          unit, scientific, integration, blindness, lock, guardrail and eval tests
```

## Scientific claim

The correct demo conclusion is:

> ExoSwarm independently recovered the transit-like signal and found that the planetary
> interpretation survived its implemented vetting. Only afterward was the external catalog status
> revealed.

Photometric and limited centroid vetting do not constitute planet confirmation and cannot exclude
every blend, stellar companion, or systematic scenario.
