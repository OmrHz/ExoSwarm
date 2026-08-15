from __future__ import annotations

from pathlib import Path

import pytest

from exoswarm.agents.provider import UnavailableProvider
from exoswarm.config import Settings
from exoswarm.domain.models import (
    ExperimentType,
    InterpretationCode,
    ScientificDisposition,
)
from exoswarm.runtime.director import ScientificDirector
from exoswarm.runtime.targets import load_demo_vault
from exoswarm.science import ScienceToolbox


@pytest.mark.parametrize(
    ("target", "expected_adaptive", "expected_disposition"),
    [
        (
            "TARGET-X17",
            ExperimentType.CENTROID_LOCALIZATION,
            ScientificDisposition.PLANETARY_INTERPRETATION_SURVIVES_VETTING,
        ),
        (
            "TARGET-X42",
            ExperimentType.HARMONIC_TEST,
            ScientificDisposition.PLANETARY_INTERPRETATION_WEAK,
        ),
    ],
)
def test_full_director_runs_real_science_and_branches_by_evidence(
    repository_root: Path,
    tmp_path: Path,
    target: str,
    expected_adaptive: ExperimentType,
    expected_disposition: ScientificDisposition,
) -> None:
    data_dir = repository_root / "data"
    runs_root = tmp_path / "runs"
    settings = Settings(
        provider="offline",
        model="none",
        api_base="https://example.invalid/v1",
        api_key=None,
        data_dir=data_dir,
        runs_dir=runs_root,
        max_agent_turns=4,
        experiment_budget=14,
        request_timeout_seconds=5,
    )
    vault, _ = load_demo_vault(data_dir)
    director = ScientificDirector(
        settings=settings,
        toolbox=ScienceToolbox(data_dir / "tess", runs_root),
        vault=vault,
        provider=UnavailableProvider("offline integration test"),
    )
    outcome = director.investigate(target, reveal=False, policy="adaptive")
    assert outcome.state.final_disposition is expected_disposition
    assert expected_adaptive in outcome.state.completed_tests
    assert outcome.receipt.sha256
    assert (outcome.run_directory / "result.json").is_file()
    assert (outcome.run_directory / "result.json.sha256").is_file()
    assert not (outcome.run_directory / "reveal.json").exists()
    assert outcome.state.experiment_budget.used <= outcome.state.experiment_budget.limit
    if target == "TARGET-X42":
        assert abs(outcome.state.candidates[0].period_days - 3.0319175718) < 0.001
        harmonic = next(
            item
            for item in outcome.ledger.items
            if item.experiment_type is ExperimentType.HARMONIC_TEST
        )
        assert harmonic.interpretation_code is InterpretationCode.PREFERRED_DOUBLE_PERIOD
