# ExoSwarm specification extract

This implementation note condenses the complete 64-page source specification. The PDF remains
authoritative; this file records the scope decisions used for the hackathon build.

## Core concept and non-negotiable boundary

ExoSwarm is mission control for a falsifiable investigation of transit-like TESS signals. AI
agents choose which permitted scientific question to ask next. Versioned deterministic Python
tools operate on real observations, create measurements and uncertainties/tolerances, and append
evidence. Agents may interpret recorded evidence but may not manufacture measurements or treat
model confidence as astrophysical probability.

The defensible novelty is the combination of adaptive adversarial experiment selection,
deterministic evidence, bounded Critic review, a visible investigation state, and a provably
blinded lock-before-reveal protocol. It is not a new transit-detection algorithm and it is not a
professional planet-confirmation system.

## Scientific workflow

1. Map the real source privately to an opaque investigation ID and load cached TESS products.
2. Inspect quality, normalize/clean, and select from a small set of defensible detrending choices.
3. Run Box Least Squares and deterministically measure period, epoch, duration, depth, signal
   quality, event count, and uncertainty or an explicit resolution/tolerance.
4. Create the planetary and non-planetary hypotheses.
5. Code-enforce signal quality, odd/even, secondary-event, and basic contamination checks.
6. Have the Skeptic choose an unused discriminating experiment from a bounded registry.
7. Have the Critic approve, request one revision, or veto; never allow recursive debate.
8. Execute the validated deterministic tool and append its typed result to the Evidence Ledger.
9. Apply declared deterministic evidence-update rules, evaluate stopping criteria, and repeat
   within experiment and turn budgets.
10. Serialize and hash the final result, lock it, then and only then enable identity/catalog reveal.

## Competing explanations

- H1: planetary transit
- H2: eclipsing binary
- H3: background or neighboring contaminant
- H4: stellar variability
- H5: instrumental/systematic artifact
- H6: period alias or harmonic

Hypothesis states and any heuristic evidence weights are declared and auditable. They are not
displayed as calibrated probabilities.

## Agent architecture

The Scientific Director is deterministic application code that owns state, budgets, permissions,
transitions, failure recovery, locking, and the bounded loop. The Skeptic identifies the strongest
remaining non-planetary explanation and selects one discriminating experiment. The Critic has the
separate objective of rejecting redundant, impermissible, or non-discriminating proposals. Each
model call receives a compact evidence packet and a strict output schema; raw time series, FITS
arrays, full histories, real identities, and catalog truth never enter model context.

Invalid structured output receives one repair attempt, followed by a safe deterministic fallback
or stop. Every failure and fallback is traced.

## Diagnostics

Mandatory for a viable candidate:

- minimum signal-quality gate;
- odd/even transit comparison;
- phase-0.5 secondary-event test;
- basic contamination screening.

Available adaptive experiments include P/2, P, and 2P harmonic/alias testing; a genuine
pixel/centroid localization diagnostic; and a second allowed preprocessing configuration when
variability or preprocessing sensitivity warrants it. Different evidence, not hidden target names,
must produce different branches.

## Evidence Ledger and scientific contracts

Every tool returns a typed `ScientificResult`, including status, experiment/tool/version,
parameters, numerical results, uncertainties or tolerances, quality flags, a bounded
interpretation code, limitations, artifact references, and provenance. Failed preconditions are
typed results with useful alternatives.

Every accepted result becomes an append-only `EvidenceItem` with input/output artifact IDs and
hashes, request/review references, and a timestamp. The ledger is the only source for scientific
numbers shown to agents or the UI. A numeric-provenance guard rejects agent text containing an
unsupported measurement.

## Blindness and result lock

Before `RESULT_LOCKED`, scientific/agent code receives only the opaque ID and has no callable or
importable catalog capability. The private mapping is used solely by the data adapter. Tests prove
that identity and catalog truth are absent from compact packets and that catalog access fails.

At finalization, ExoSwarm writes the pre-reveal result, computes its SHA-256 hash, records the lock
event, and makes that result immutable. A separate reveal artifact may be created only afterward.
The trace records the exact capability transition.

## Evaluation and demonstration

At minimum the same runtime investigates one real known planet and one real eclipsing-binary or
false-positive control. They must show different evidence, adaptive choices, trajectories, and
dispositions. Graders enforce blindness, period tolerance, mandatory diagnostics, valid schemas,
budgets, non-planetary handling, unsupported-number rejection, and trajectory diversity without
requiring one arbitrary exact path. A fixed-checklist baseline is compared honestly with the
adaptive selector if the core is stable.

The UI is a mission-control view, not a chat interface. It shows the opaque target and catalog gate,
real light curves and scientific artifacts, hypothesis state, Skeptic and Critic decisions, the
Evidence Ledger, a dramatic lock event, and the post-lock catalog comparison. All numbers link back
to deterministic evidence.

## Priority decision

### P0 — must ship

- cached real TESS light curve pipeline through BLS and phase folding;
- two suitable targets and differing evidence-driven paths;
- typed state, hypotheses, tool contracts, Evidence Ledger, and traces;
- bounded adaptive selection, budgets, stopping rules, and safe fallback;
- mandatory vetting and implemented harmonic/centroid tools;
- opaque IDs, package/capability isolation, result hash, post-lock reveal, and blindness tests;
- no fake confidence and precise claim language;
- offline deterministic reproduction of the showcased investigations.

### P1 — high-value polish

- genuine centroid/pixel output with uncertainty/significance limits;
- visible Critic and hypothesis updates;
- numeric-provenance enforcement;
- adaptive-versus-fixed mini-ablation and repeated consistency checks;
- information-value and token/latency tracing;
- polished mission-control UI and trace viewer.

### P2 — deliberately deferred

- model routing or escalation;
- extra agents or broad target coverage;
- multi-sector stitching;
- sophisticated transit fits or probabilistic validation;
- many additional diagnostics or distributed infrastructure.

## Permitted claim language

Use: “ExoSwarm independently recovered the transit-like signal and found that the planetary
interpretation survived its implemented photometric vetting. Only afterward was the external
catalog status revealed.”

Do not claim autonomous planet discovery or confirmation. NASA/catalog confirmation remains an
external fact, separate from ExoSwarm's locked categorical disposition. Always state that limited
photometric and centroid vetting cannot exclude every astrophysical false-positive scenario and
does not replace follow-up observations or professional validation.

