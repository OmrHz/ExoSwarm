from __future__ import annotations

from exoswarm.domain.models import InterpretationCode, ScientificResult
from exoswarm.science import candidate_from_search_result


def test_planet_candidate_is_recovered_with_honest_measurements(prepared_cases) -> None:
    result = prepared_cases["TARGET-X17"].search
    assert result.interpretation_code is InterpretationCode.DETECTED
    values = result.numerical_results
    assert abs(values["period_days"] - 3.739494) < 0.001
    assert 12_000 < values["transit_depth_ppm"] < 18_000
    assert 1.5 < values["duration_hours"] < 2.3
    assert values["signal_to_noise"] > 50
    assert values["observed_events"] >= 4
    assert values["initial_alias_selected"] == 0
    assert result.uncertainties["period_days"].kind == "tolerance"
    assert result.uncertainties["duration_hours"].kind == "resolution"
    assert {item.role for item in result.output_artifacts} == {
        "bls_periodogram_data",
        "bls_periodogram_plot",
    }


def test_eb_initial_event_cadence_is_explicitly_flagged_alias(prepared_cases) -> None:
    result = prepared_cases["TARGET-X42"].search
    values = result.numerical_results
    assert abs(values["period_days"] - 1.51596) < 0.001
    assert abs(values["bls_global_period_days"] - 3.031918) < 0.002
    assert values["initial_alias_selected"] == 1
    assert values["half_period_snr_ratio"] > 0.95
    assert values["transit_depth_ppm"] > 50_000
    assert any(flag.code == "NEAR_DEGENERATE_HALF_PERIOD" for flag in result.quality_flags)


def test_candidate_adapter_uses_shared_domain_schema(prepared_cases) -> None:
    result: ScientificResult = prepared_cases["TARGET-X17"].search
    candidate = candidate_from_search_result(result)
    assert candidate.period_days == result.numerical_results["period_days"]
    assert candidate.epoch_btjd == result.numerical_results["epoch_btjd"]
    assert candidate.uncertainties["period_days"] == result.uncertainties["period_days"]
