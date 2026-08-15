from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from exoswarm.domain.models import (
    BackendTargetMappingRef,
    ExperimentType,
    InvestigationState,
    ScientificResult,
    ToolRequest,
)
from exoswarm.science import ScienceToolbox, candidate_from_search_result


@dataclass
class PreparedCase:
    toolbox: ScienceToolbox
    state: InvestigationState
    search: ScientificResult


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def prepared_cases(
    repository_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, PreparedCase]:
    runs_root = tmp_path_factory.mktemp("science-runs")
    toolbox = ScienceToolbox(repository_root / "data" / "tess", runs_root)
    cases: dict[str, PreparedCase] = {}
    for target in ("TARGET-X17", "TARGET-X42"):
        state = InvestigationState(
            opaque_target_id=target,
            backend_target_mapping=BackendTargetMappingRef(
                mapping_key=f"science-test-mapping-{target.lower()}"
            ),
        )
        for experiment in (ExperimentType.NORMALIZATION, ExperimentType.DETRENDING):
            result = toolbox.execute(
                ToolRequest(experiment_type=experiment, requested_by="pytest"), state
            )
            assert isinstance(result, ScientificResult), result
        search = toolbox.execute(
            ToolRequest(experiment_type=ExperimentType.TRANSIT_SEARCH, requested_by="pytest"),
            state,
        )
        assert isinstance(search, ScientificResult), search
        state.candidates.append(candidate_from_search_result(search))
        cases[target] = PreparedCase(toolbox=toolbox, state=state, search=search)
    return cases
