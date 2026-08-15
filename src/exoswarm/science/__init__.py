"""Deterministic astronomy tools for ExoSwarm.

Agents choose which registered operation to request; this package alone performs
the numerical measurements and returns the shared typed domain contracts.
"""

from .common import ArtifactStore, ScienceManifest, load_science_manifest
from .data import (
    detrend_light_curve,
    inspect_light_curve_quality,
    load_cached_observation,
    load_spoc_light_curve,
    normalize_and_clean_light_curve,
)
from .pixels import centroid_localization
from .toolbox import ScienceToolbox, candidate_from_search_result
from .transit import compare_detrending_sensitivity, phase_fold_candidate, search_transits
from .vetting import (
    assess_signal_quality,
    contamination_screen,
    harmonic_alias_test,
    odd_even_test,
    secondary_eclipse_test,
)

__all__ = [
    "ArtifactStore",
    "ScienceManifest",
    "ScienceToolbox",
    "assess_signal_quality",
    "candidate_from_search_result",
    "centroid_localization",
    "compare_detrending_sensitivity",
    "contamination_screen",
    "detrend_light_curve",
    "harmonic_alias_test",
    "inspect_light_curve_quality",
    "load_cached_observation",
    "load_science_manifest",
    "load_spoc_light_curve",
    "normalize_and_clean_light_curve",
    "odd_even_test",
    "phase_fold_candidate",
    "search_transits",
    "secondary_eclipse_test",
]
