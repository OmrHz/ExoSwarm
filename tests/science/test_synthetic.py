from __future__ import annotations

from pathlib import Path

import numpy as np

from exoswarm.domain.models import (
    Candidate,
    ExperimentType,
    InterpretationCode,
    ScientificResult,
    ScientificStatus,
)
from exoswarm.science import (
    ArtifactStore,
    ScienceManifest,
    assess_signal_quality,
    compare_detrending_sensitivity,
    search_transits,
)


def _synthetic_manifest() -> ScienceManifest:
    product = {
        "path": "unused.fits",
        "size_bytes": 1,
        "sha256": "0" * 64,
        "media_type": "application/fits",
    }
    return ScienceManifest.model_validate(
        {
            "schema_version": "1.0",
            "opaque_target_id": "TARGET-SYNTHETIC",
            "mission": "TESS",
            "pipeline": "SPOC",
            "sector": 1,
            "cadence_seconds": 120.0,
            "retrieved_at": "2026-08-14T00:00:00Z",
            "archive_collection_url": "https://archive.stsci.edu/missions-and-data/tess/data-products",
            "products": {"light_curve": product, "target_pixel": product},
            "neighbor_context": None,
            "notes": ["Synthetic deterministic unit-test manifest; no source product is opened."],
        }
    )


def test_deterministic_bls_recovers_synthetic_period(tmp_path: Path) -> None:
    rng = np.random.default_rng(73)
    time = np.arange(0.0, 22.0, 2.0 / 1440.0)
    period = 2.75
    epoch = 0.43
    duration = 0.12
    phase_time = (((time - epoch + 0.5 * period) % period) / period - 0.5) * period
    flux = np.ones_like(time)
    flux[np.abs(phase_time) <= duration / 2] -= 0.012
    flux += rng.normal(0, 0.0015, time.size)
    flux_err = np.full_like(time, 0.0015)
    store = ArtifactStore(tmp_path / "run", "TARGET-SYNTHETIC")
    store.save_npz(
        "cleaned_light_curve",
        "cleaned_light_curve_data",
        time_btjd=time,
        flux=flux,
        flux_err=flux_err,
        trend=np.ones_like(time),
    )
    result = search_transits(
        _synthetic_manifest(),
        store,
        min_period_days=0.7,
        max_period_days=5.0,
        min_duration_hours=1.0,
        max_duration_hours=5.0,
        frequency_samples=6000,
    )
    assert isinstance(result, ScientificResult)
    assert result.interpretation_code is InterpretationCode.DETECTED
    assert abs(result.numerical_results["period_days"] - period) < 0.003
    assert result.numerical_results["transit_depth_ppm"] > 9_000
    assert result.numerical_results["observed_events"] >= 7


def test_low_snr_candidate_remains_failed_screen_not_fake_probability() -> None:
    candidate = Candidate(
        period_days=3.0,
        epoch_btjd=100.0,
        transit_depth_ppm=500.0,
        duration_hours=2.0,
        signal_to_noise=4.2,
        observed_events=3,
    )
    result = assess_signal_quality(_synthetic_manifest(), candidate)
    assert result.interpretation_code is InterpretationCode.FAIL
    assert result.numerical_results["passes_minimum_quality"] == 0
    assert all("probability" not in key for key in result.numerical_results)


def test_alternate_detrending_flags_candidate_that_disappears(tmp_path: Path) -> None:
    rng = np.random.default_rng(129)
    time = np.arange(0.0, 12.0, 2.0 / 1440.0)
    period = 2.4
    epoch = 0.7
    duration = 0.10
    phase_time = (((time - epoch + 0.5 * period) % period) / period - 0.5) * period
    nominal_flux = np.ones_like(time)
    nominal_flux[np.abs(phase_time) <= duration / 2] -= 0.01
    nominal_flux += rng.normal(0, 0.001, time.size)
    alternate_flux = 1.0 + rng.normal(0, 0.001, time.size)
    flux_err = np.full_like(time, 0.001)
    store = ArtifactStore(tmp_path / "run", "TARGET-SYNTHETIC")
    store.save_npz(
        "normalized_light_curve",
        "normalized_light_curve_data",
        time_btjd=time,
        flux=nominal_flux,
        flux_err=flux_err,
    )
    store.save_npz(
        "cleaned_light_curve",
        "cleaned_light_curve_data",
        time_btjd=time,
        flux=nominal_flux,
        flux_err=flux_err,
        trend=np.ones_like(time),
    )
    store.save_npz(
        "alternate_cleaned_light_curve",
        "alternate_cleaned_light_curve_data",
        time_btjd=time,
        flux=alternate_flux,
        flux_err=flux_err,
        trend=np.ones_like(time),
    )
    preprocessing = ScientificResult(
        status=ScientificStatus.SUCCESS,
        experiment_type=ExperimentType.ALTERNATE_DETRENDING,
        tool_name="synthetic.alternate_detrending",
        tool_version="test",
        parameters={"method": "synthetic-removal", "window_days": 1.5},
        interpretation_code=InterpretationCode.PROCESSED,
    )
    candidate = Candidate(
        period_days=period,
        epoch_btjd=epoch,
        transit_depth_ppm=10_000,
        duration_hours=duration * 24,
        signal_to_noise=50,
        observed_events=5,
    )
    result = compare_detrending_sensitivity(_synthetic_manifest(), store, candidate, preprocessing)
    assert isinstance(result, ScientificResult)
    assert result.interpretation_code is InterpretationCode.PREPROCESSING_SENSITIVE
    assert result.numerical_results["passes_preprocessing_robustness"] == 0
    assert result.numerical_results["alternate_to_nominal_snr_ratio"] < 0.7
