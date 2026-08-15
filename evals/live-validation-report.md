# Featherless live-model validation

Validated on 2026-08-15 against six fresh investigations. This report was generated from the
hash-chained traces, Evidence Ledgers, locked results, and separate reveal artifacts under
`runs/live-validation/20260815-featherless-v4/`. The API credential is not stored in this report
or any run artifact.

## Provider

- Provider: `featherless`
- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Authentication: PASS
- Live inference: PASS
- Skeptic structured output: PASS
- Critic structured output: PASS

The model/provider combination consistently returned a complete JSON object body without its
opening `{`. ExoSwarm's narrowly bounded adapter repair restored that one character and still
required strict JSON decoding, no trailing prose, and complete Pydantic validation. Every campaign
decision is therefore classified `REPAIRED_LIVE_MODEL`; none is reported as first-pass valid.

Campaign reliability statistics below deliberately cover the six retained investigations. Before
the fix, nine synthetic smoke structured requests made 16 live completions: seven requests exhausted
their then-available model repair and used the explicitly labeled deterministic fallback. After the
parser fix, the final two-role smoke passed as repaired-live, and the six-run campaign added 12 more
live completions with zero fallback. Thus the full development session made 28 provider completions;
no failed smoke was counted as a successful validation run.

## TARGET-X17 — three runs

| Trial | Run ID | Skeptic choice | Critic | Executed | Repairs | Deterministic fallbacks | Latency | Tokens | Disposition | Lock/reveal |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 01 | `TRACE-676AA6C5AE8D490C8BABA8CFBC851C02` | H3 → `centroid_localization` | APPROVE | `centroid_localization` | 2 | 0 | 10.154 s | 11,128 | PLANETARY INTERPRETATION SURVIVES IMPLEMENTED VETTING | PASS/PASS |
| 02 | `TRACE-1AD5BD5A034349D1B4D111C0A2929869` | H3 → `centroid_localization` | APPROVE | `centroid_localization` | 2 | 0 | 10.200 s | 11,119 | PLANETARY INTERPRETATION SURVIVES IMPLEMENTED VETTING | PASS/PASS |
| 03 | `TRACE-5D422B61C6AA41EB96AAEBE50B8EE9A4` | H3 → `centroid_localization` | APPROVE | `centroid_localization` | 2 | 0 | 11.750 s | 11,067 | PLANETARY INTERPRETATION SURVIVES IMPLEMENTED VETTING | PASS/PASS |

All three Skeptics grounded the choice in the detected nearby-source evidence. All three Critics
found the request permitted, unused, precondition-valid, and discriminating. Deterministic pixel
analysis returned `TARGET_CONSISTENT` with a difference-source offset of 0.043248 pixel.

## TARGET-X42 — three runs

| Trial | Run ID | Skeptic choice | Critic | Executed | Repairs | Deterministic fallbacks | Latency | Tokens | Disposition | Lock/reveal |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 01 | `TRACE-5DB80B69513B4F128A5FE47597D5FD73` | H6 → `harmonic_test` | APPROVE | `harmonic_test` | 2 | 0 | 11.879 s | 11,576 | PLANETARY INTERPRETATION WEAK | PASS/PASS |
| 02 | `TRACE-F34F89776B984C4494EEEABE48363AAB` | H6 → `harmonic_test` | APPROVE | `harmonic_test` | 2 | 0 | 11.294 s | 11,366 | PLANETARY INTERPRETATION WEAK | PASS/PASS |
| 03 | `TRACE-160B63EC65434EAC96025CC86503E29E` | H6 → `harmonic_test` | APPROVE | `harmonic_test` | 2 | 0 | 11.527 s | 11,444 | PLANETARY INTERPRETATION WEAK | PASS/PASS |

All three Skeptics grounded the choice in the extreme odd/even mismatch and recorded half-period
alias flag. All three Critics approved the bounded P/2–P–2P comparison. Deterministic harmonic
analysis selected the double-period interpretation and weakened the planetary disposition.

## Consistency and reliability

- X17 scientifically valid trajectories: 3/3
- X42 scientifically valid trajectories: 3/3
- Overall live-model valid trajectories: 6/6
- Exact adaptive experiment consistent within X17: yes (3/3 centroid)
- Exact adaptive experiment consistent within X42: yes (3/3 harmonic)
- Different evidence changed the cross-target trajectory: PASS
- Independent trace IDs and run directories: PASS
- Total structured requests: 12
- Total live HTTP completions: 12
- First-pass schema-valid responses: 0/12 (0%)
- Repair attempts: 12 (100%)
- Successful repairs: 12/12 (100%)
- Failed repairs: 0
- Deterministic fallbacks: 0 (0%)
- Invalid experiment requests: 0
- Redundant experiment requests: 0
- Unsupported numeric prose claims attempted and mechanically repaired: 3
- Unsupported numeric claims remaining after sanitation: 0
- Token usage: 64,107 prompt + 3,593 completion = 67,700 total

The three provenance interventions were one X42 trial-01 claim and two X17 trial-02 claims. They
changed explanatory prose only; no measurement, tool request, or scientific result was supplied by
the model.

## Deterministic measurements

The values were identical across all three runs of each target.

| Measurement | Live deterministic result | Existing reference | Difference |
|---|---:|---:|---:|
| X17 period | 3.739466864836875 d | 3.739466865 d | 0.000000000163125 d |
| X17 depth | 15,050.928162 ppm | 15,051 ppm | 0.071838 ppm |
| X17 S/N | 105.157254 | 105.2 | 0.042746 |
| X42 initial period | 1.515946218709595 d | 1.515946219 d | 0.000000000290405 d |
| X42 resolved period | 3.031910459977699 d | 3.031910460 d | 0.000000000022301 d |

Additional locked measurements: X17 epoch 1632.090746305 BTJD, duration 1.938313 h, four events;
X42 resolved epoch 1660.669716786 BTJD, depth 135,486.145169 ppm, duration 3.656853 h, S/N
2269.158469, seven events. These are deterministic Evidence Ledger values, not model estimates.

## Blindness and integrity

- Target identity hidden from every pre-lock agent request: PASS
- Catalog access blocked before `RESULT_LOCKED`: PASS
- Known catalog parameters absent from pre-lock context: PASS
- Context-key preflight and SHA-256 digest recorded for all 12 requests: PASS
- Result locked before catalog access and reveal: PASS (6/6)
- Stored result SHA-256 verified: PASS (6/6)
- Reveal stored separately from locked result: PASS (6/6)
- Locked result remained byte-identical after reveal: PASS (6/6)
- Trace and Evidence Ledger hash commitments verified: PASS (6/6)

Each post-lock reveal matched the pinned expected catalog identity and status for its curated case.
Those identity-bearing values remain in the separately gated reveal artifacts rather than this
pre-demo documentation. Catalog status remains separate from ExoSwarm's own photometric
disposition.

## Mission Control

- TARGET-X17 live-agent locked and revealed states: PASS
- TARGET-X42 live-agent locked and revealed states: PASS
- Eight real Plotly science charts and fifteen tables per revealed target: PASS
- Live decision source, provider, model, request IDs, Skeptic/Critic reasoning, and divergent
  adaptive branches visible: PASS
- Pre-lock frontend identity scan and all-state credential scan: PASS
- Exact `exoswarm.cli ui` launch and Streamlit health endpoint: PASS

## Regression results

- Full pytest suite: 123 passed
- Ruff lint: PASS
- Ruff format: PASS after formatting the two modified modules
- Dependency validation (`pip check`): PASS
- Four cached FITS SHA-256/size checks: PASS
- Blindness, import-boundary, numeric-provenance, and result-lock focused suite: 19 passed
- Streamlit UI suite: 17 passed
- Fresh offline reproduction: PASS for both targets; cross-case diversity PASS
- Adaptive-vs-fixed evaluation: PASS; equal experiment counts reported honestly, and only the
  adaptive X42 trajectory selected the needed harmonic test and reached the weak disposition

## Problems found and fixes

1. Featherless JSON mode returned object bodies missing one opening brace. The adapter now requests
   official JSON mode and permits only that observed one-character normalization before strict JSON
   and schema validation; the repair is explicit in traces.
2. Decision origin was previously implicit. It is now recorded and displayed as `LIVE_MODEL`,
   `REPAIRED_LIVE_MODEL`, or `DETERMINISTIC_FALLBACK`; `--require-live` fails closed for validation.
3. Agents saw experiment names but not exact bounded parameter contracts. Identity-safe contracts
   are now included, while candidate IDs and harmonic ephemerides are deterministically bound and
   registry-validated.
4. The Critic prompt did not explicitly require exact request-ID copying. It now does, and the
   runtime still rejects a mismatch.
5. Agent-request traces lacked replay-safe context proof. They now record a packet digest, evidence
   IDs, available/completed tests, lock state, and Critic proposal metadata after a strict blindness
   preflight—never the full prompt or raw arrays.
6. Provider failures were too coarsely categorized and `Settings` repr could expose a key. Safe
   status/category diagnostics and `repr=False` secret handling were added and tested.
7. Mission Control did not positively prove live provenance. Skeptic and Critic cards now show the
   trace-backed decision source, provider, model, attempts, and provider request IDs.

## Demo artifacts

Best paired demo root: `runs/live-validation/20260815-featherless-v4/trial-01/`.

- X17 trace: `runs/live-validation/20260815-featherless-v4/trial-01/TARGET-X17/trace.jsonl`
- X17 ledger/result/reveal: adjacent `evidence.jsonl`, `result.json`, `result.json.sha256`, and
  `reveal.json`
- X42 trace: `runs/live-validation/20260815-featherless-v4/trial-01/TARGET-X42/trace.jsonl`
- X42 ledger/result/reveal: adjacent `evidence.jsonl`, `result.json`, `result.json.sha256`, and
  `reveal.json`

Launch the replay UI with:

```powershell
.\.venv\Scripts\python.exe -m exoswarm.cli ui --runs-root runs/live-validation/20260815-featherless-v4/trial-01
```
