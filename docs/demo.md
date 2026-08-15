# Three-minute demo flow

This script is designed for a recorded submission. Use large burned-in captions; do not begin with
slides or a chat window.

## Before recording

Choose a fresh run root for every take; the runtime deliberately refuses to overwrite an existing
locked investigation. Use the same value in both PowerShell terminals below.

Terminal 1 stages two real, locked-but-unrevealed investigations and then opens Mission Control:

```powershell
$demoRuns = "runs\video-demo-001" # Change the suffix for each take.
.\.venv\Scripts\python.exe -m exoswarm.cli verify-cache
.\.venv\Scripts\python.exe -m pytest tests\security -q
.\.venv\Scripts\python.exe -m exoswarm.cli run TARGET-X17 --offline --runs-root $demoRuns
.\.venv\Scripts\python.exe -m exoswarm.cli run TARGET-X42 --offline --runs-root $demoRuns
.\.venv\Scripts\python.exe -m exoswarm.cli ui --runs-root $demoRuns
```

For a genuinely model-directed trace, configure the Featherless key, omit `--offline`, and add
`--require-live` to both run commands. The command then fails validation rather than silently
counting a deterministic fallback as live success. Mission Control shows the exact decision source,
provider, model, attempts, and provider request IDs. Keep a verified offline run available as a
failure-safe demo, but describe it accurately as the declared deterministic policy fallback.

Mission Control is a read-only audit surface. At the reveal beat, use Terminal 2 to cross the
ground-truth gate, then click **Refresh artifacts** in Mission Control (or enable **Follow artifact
updates** beforehand):

```powershell
$demoRuns = "runs\video-demo-001" # Must match Terminal 1.
.\.venv\Scripts\python.exe -m exoswarm.cli reveal TARGET-X17 --runs-root $demoRuns
```

## 0:00–0:15 — Hook

Open Mission Control on `TARGET-X17` before reveal.

On screen:

```text
UNKNOWN TESS TARGET — TARGET-X17
GROUND TRUTH — LOCKED
NASA PARAMETERS — UNAVAILABLE
```

Narration:

> This is real cached NASA TESS data. ExoSwarm has not been given the target identity or known
> planetary parameters. Its job is not merely to find a dip—it must seek evidence that could make
> its own planetary interpretation fail.

## 0:15–0:50 — Deterministic search

Show the source-hash verification, data-quality card, raw and cleaned light curves, BLS
periodogram, folded curve, and candidate measurement cards. Point to the evidence/tool provenance
label under the measurements.

Caption:

```text
ASTROPY / NUMPY / SCIPY PRODUCED THESE MEASUREMENTS
THE MODEL DID NOT MEASURE THE PERIOD OR DEPTH
```

## 0:50–1:20 — The adaptive decision

Show the mandatory odd/even and secondary checks, then the contamination evidence. Keep the
Skeptic and Critic cards in view:

```text
STRONGEST ALTERNATIVE — BACKGROUND / NEIGHBOR CONTAMINANT
SKEPTIC REQUEST — CENTROID LOCALIZATION
CRITIC — APPROVE
```

Show the cached target-pixel out-of-transit, in-transit, and difference images plus the measured
centroid result. Then show the Evidence Ledger append and hypothesis-state change.

Narration:

> The agent did not calculate the centroid. It chose spatial localization because the current
> evidence made that experiment more discriminating than another light-curve check. Deterministic
> pixel code then produced and recorded the result.

## 1:20–1:50 — Negative control

Switch to `TARGET-X42` and keep its opaque identity locked. Show the initial half-period event
cadence and the catastrophic odd/even depth difference.

```text
SKEPTIC REQUEST — TEST P/2, P, 2P
CRITIC — APPROVE
RESULT — 2P SEPARATES PRIMARY AND SECONDARY EVENTS
DISPOSITION — PLANETARY INTERPRETATION WEAK
```

Narration:

> The same policy does not always end at planet. Different evidence selected a different
> experiment, resolved an orbital-period alias, and produced a different disposition.

## 1:50–2:05 — Prove the boundary

Briefly show:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\security -q
```

Zoom in on the exact catalog-before-lock, target-identity isolation, immutable lock, and static
import-boundary tests.

Caption:

```text
BLINDNESS IS A SOFTWARE INVARIANT, NOT A PROMPT PROMISE
```

## 2:05–2:35 — Commit, then reveal

Return to X17. Show the final ExoSwarm disposition and limitations while identity remains locked.
Show the verified result-lock panel and its commit event:

```text
SERIALIZING RESULT.JSON…
SHA-256 COMMIT MARKER WRITTEN
RESULT LOCKED — READ ONLY
```

Only then run the documented Terminal 2 reveal command and refresh the audit surface. Compare the
locked measurement with the external catalog record and say explicitly:

> ExoSwarm's implemented photometric vetting did not confirm this planet. Only after ExoSwarm
> locked its independent measurements did the external catalog reveal the target identity and
> catalog status shown here.

## 2:35–2:52 — Evaluate the agent claim

Show the generated constraint report and adaptive-versus-fixed table. Do not cherry-pick or alter
equal metrics.

Caption:

```text
WE GRADE CONSTRAINTS, BRANCHES, DISPOSITIONS, FAILURES, COST AND CONSISTENCY
NOT HOW INTELLIGENT THE EXPLANATION SOUNDS
```

## 2:52–3:00 — Close

End on the hash, ledger root, and two different branch traces.

> ExoSwarm measured a real signal, chose evidence to challenge its current explanation, locked the
> result, and only then saw the catalog. The AI does not replace the scientific tools. It decides
> which scientific question to ask next.
