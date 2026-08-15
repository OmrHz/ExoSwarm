# Deterministic science methods

ExoSwarm's scientific layer is deliberately smaller than a professional TESS validation
pipeline. It is sufficient for a reproducible hackathon investigation, but it does not claim
planet confirmation or calibrated false-positive probabilities.

## Cached observations and provenance

The two demonstration cases each use one public, unmodified, two-minute SPOC light-curve FITS
file and one matching SPOC target-pixel FITS file. The four files total 102,584,960 bytes
(97.83 MiB). Their sizes and SHA-256 digests are recorded in identity-free manifests under
`data/tess/TARGET-*/science_manifest.json`. Exact TIC-bearing MAST product URLs and external
catalog truth are kept in the backend-private `data/private/targets.json`; the science package
does not import the catalog/security layer.

The public science result cites the MAST/TESS collection and cached source hash without exposing
the original TIC-bearing filename. Generated UI artifacts contain arrays only, never raw FITS
headers such as `OBJECT` or `TICID`.

## Quality, normalization, and cleaning

The light-curve loader reads `TIME`, `PDCSAP_FLUX`, `PDCSAP_FLUX_ERR`, and `QUALITY` from a
SPOC FITS product. Quality inspection reports usable cadence counts, observing baseline, median
cadence, gaps larger than 0.25 days, and approximate duty cycle.

Normalization retains finite, positive-flux cadences with `QUALITY == 0`, divides by the median
PDCSAP flux, and removes only the bright tail above eight robust MAD sigmas. It never applies
symmetric negative sigma clipping: transit and eclipse dimmings are therefore preserved. This is
tested directly for the real cached cases.

PDCSAP is already a deterministic SPOC systematics-corrected product. ExoSwarm then permits only
two limited, gap-aware detrending methods:

- a 0.5–3 day running-median division (24 hours in the mandatory path); or
- a transit-protected Savitzky–Golay sensitivity run. Potential dimmings and bright excursions
  are excluded from this smoother's training series, but the original cadences remain in its
  output.

The point-to-point MAD is reported as a noise proxy. It is not a red-noise model.

The adaptive alternate-detrending experiment does more than create another curve. It performs
the same local Astropy BLS re-search around the candidate on both the nominal and alternate
products, then compares period, depth, S/N, and event count. It reports `ROBUST` only when the
period shift is at most 0.5%, the depth change is at most 25% or 5 combined formal sigmas, the
alternate-to-nominal S/N ratio is at least 0.70, alternate S/N is at least 7, and at least two
events remain. Otherwise it reports `PREPROCESSING_SENSITIVE`. These are declared heuristic
robustness thresholds, not probabilities; the local comparison does not search for an unrelated
dominant period elsewhere in the full period range.

## Transit search and measurements

Transit search uses `astropy.timeseries.BoxLeastSquares` over a bounded frequency grid. The
mandatory runtime uses 0.5–15 days and trial durations of 1, 2, 3, 4, and 6 hours; the effective
maximum is capped at half the observed baseline so at least two events can occur. A local grid
refines the strongest peak.

A declared, target-independent harmonic tie rule handles alternating eclipses conservatively. If
the P/2 peak retains at least 96% of the strongest BLS depth S/N, ExoSwarm records the shorter
event cadence as a provisional ephemeris and flags it for odd/even plus harmonic resolution. This
causes the negative control to begin at approximately 1.516 days without using its hidden name or
catalog period.

When no near-degenerate half-period family is present, a robust lightweight trapezoid fit refines
period, epoch, total duration, and depth. This is not a limb-darkened physical transit fit. The
reported approximate radius ratio is `sqrt(depth)` and explicitly neglects dilution, limb
darkening, and grazing geometry.

Uncertainty fields distinguish standard uncertainties from numerical resolution/tolerance:

- period: maximum of the local frequency-grid resolution and cadence divided by the measured
  event baseline;
- epoch: half a cadence;
- duration: maximum of one cadence and half the local duration-grid spacing;
- depth: robust trapezoid residual standard error, or the Astropy BLS formal error for a
  provisionally aliased box;
- S/N: BLS depth S/N using SPOC per-cadence errors, not a false-alarm probability.

## Mandatory diagnostics

Minimum signal-quality screening requires S/N at least 7, at least two observed events, positive
depth below 500,000 ppm, and duration no larger than 20% of the period. These are declared MVP
rules, not calibrated planet probabilities.

The odd/even test requires at least four usable events with at least two of each parity. Astropy
BLS computes inverse-variance odd and even depths; a difference of at least 3 sigma is marked
`INCONSISTENT`. Equal odd/even depths do not exclude an equal-depth binary.

The secondary-eclipse diagnostic measures a box at phase 0.5 and performs a bounded phase scan.
The fixed phase threshold is 5 sigma and the conservative scan threshold is 10 sigma. The scan is
not presented as a trials-corrected false-alarm probability. A P/2 ephemeris overlays primary and
secondary events at phase zero, so a null phase-0.5 result at P/2 must be interpreted with the
odd/even and 2P evidence.

The basic contamination screen records SPOC `CROWDSAP`, `FLFRCSAP`, and cached TIC neighbor
geometry. It recommends localization when estimated contaminating flux is at least 5% or a
neighbor lies within 42 arcseconds. It cannot determine which source dims.

## Harmonic and spatial diagnostics

The harmonic tool fits BLS boxes near P/2, P, and 2P. A 2P orbital interpretation is selected
only when its BLS S/N exceeds the nominal score by at least 1% and the nominal odd/even mismatch
is at least 3 sigma. A robust primary trapezoid is then fit at the selected 2P ephemeris, while a
separate phase-0.5 box records the secondary event. These declared heuristic rules are auditable;
they are not probabilistic binary validation.

When a harmonic test revises the candidate, it emits replacement period, epoch, duration, depth,
S/N, and observed-event measurements. Period and duration carry explicit tolerances and epoch
carries a half-cadence resolution so the runtime never retains uncertainty metadata from the
superseded provisional ephemeris.

The real pixel diagnostic reads every calibrated frame from the cached target-pixel FITS file. It
constructs mean out-of-transit, in-transit, and difference images; bootstraps the positive
difference-image moment centroid; and measures the transit-correlated SPOC-aperture photocenter
shift. In a crowded aperture, target dimming should move the photocenter toward a steady neighbor,
whereas neighbor dimming moves it in the opposite direction. The result is `TARGET_CONSISTENT`
only when the difference source lies within 0.5 pixel and any significant shift has the
target-dimming direction. This is a genuine single-sector pixel test, but it is not a calibrated
SPOC PRF-centroid analysis and cannot exclude unresolved companions.

## Real-case numerical regression

The offline Director integration tests recover these blinded values from the cached products:

| Opaque case | Initial period | Adaptive diagnostic | Resolved period | Photometric disposition |
| --- | ---: | --- | ---: | --- |
| TARGET-X17 | about 3.73947 d | centroid localization | unchanged | planetary interpretation survives implemented vetting |
| TARGET-X42 | about 1.51595 d (flagged P/2) | harmonic 2P test | about 3.03191 d | planetary interpretation weak |

External catalog identity and values are unavailable to science/agent code during those
measurements. The backend reveals them only after `result.json` and its SHA-256 lock exist.

## Claim boundary

The strongest permitted conclusion is that a planetary interpretation survived the implemented
photometric and centroid vetting. This does not constitute confirmation. Spectroscopy, radial
velocities, higher-resolution imaging, additional sectors, or professional statistical
validation can still be required.
