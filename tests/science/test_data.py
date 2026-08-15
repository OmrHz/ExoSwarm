from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from exoswarm.domain.models import (
    BackendTargetMappingRef,
    ExperimentType,
    InterpretationCode,
    InvestigationState,
    ScientificFailure,
    ScientificResult,
    ScientificStatus,
    ToolRequest,
)
from exoswarm.science import ScienceToolbox, load_science_manifest
from exoswarm.science import common as science_common
from exoswarm.science.common import ArtifactStore, load_npz, sha256_file


def _state(target: str) -> InvestigationState:
    return InvestigationState(
        opaque_target_id=target,
        backend_target_mapping=BackendTargetMappingRef(mapping_key="science-data-test-key-0001"),
    )


def test_science_manifests_are_identity_free_and_products_match_hashes(
    repository_root: Path,
) -> None:
    data_root = repository_root / "data" / "tess"
    forbidden = {"261867566", "80059889", "TOI-905", "actual_target_identity"}
    for target in ("TARGET-X17", "TARGET-X42"):
        path = data_root / target / "science_manifest.json"
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden)
        manifest = load_science_manifest(data_root, target)
        assert manifest.opaque_target_id == target
        for product in manifest.products.values():
            source = path.parent / product.path
            assert source.stat().st_size == product.size_bytes
            assert sha256_file(source) == product.sha256


def test_load_quality_normalization_and_negative_excursion_preservation(
    repository_root: Path, tmp_path: Path
) -> None:
    toolbox = ScienceToolbox(repository_root / "data" / "tess", tmp_path / "runs")
    state = _state("TARGET-X17")
    loaded = toolbox.execute(
        ToolRequest(experiment_type=ExperimentType.LOAD_CACHED_DATA, requested_by="pytest"),
        state,
    )
    quality = toolbox.execute(
        ToolRequest(experiment_type=ExperimentType.QUALITY_INSPECTION, requested_by="pytest"),
        state,
    )
    normalized = toolbox.execute(
        ToolRequest(experiment_type=ExperimentType.NORMALIZATION, requested_by="pytest"),
        state,
    )
    assert isinstance(loaded, ScientificResult)
    assert isinstance(quality, ScientificResult)
    assert isinstance(normalized, ScientificResult)
    assert quality.interpretation_code is InterpretationCode.ACCEPTABLE
    assert quality.numerical_results["usable_cadences"] == 12829
    assert normalized.parameters["negative_sigma_clipping"] is False
    artifact = next(item for item in normalized.output_artifacts if item.role.endswith("_data"))
    arrays = load_npz(toolbox.run_directory("TARGET-X17") / str(artifact.path))
    assert np.nanmin(arrays["flux"]) < 0.985
    assert arrays["time_btjd"].size == normalized.numerical_results["normalized_cadences"]
    assert loaded.provenance[0].software == {
        "exoswarm-science": loaded.tool_version,
        "astropy": version("astropy"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "matplotlib": version("matplotlib"),
    }


def test_deterministic_npz_hash_and_safe_run_relative_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run", "TARGET-TEST")
    first = store.save_npz("example", "example_data", z=np.arange(4), a=np.array([1.5]))
    second = store.save_npz("example", "example_data", z=np.arange(4), a=np.array([1.5]))
    assert first.sha256 == second.sha256
    assert first.path == "artifacts/science/example.npz"
    manifest_text = (tmp_path / "run" / "artifacts" / "artifacts.json").read_text()
    assert "source_lc.fits" not in manifest_text
    assert "artifacts/science/example.npz" in manifest_text


def test_artifact_manifest_retries_transient_windows_sharing_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = science_common.os.replace
    failures = 0

    def flaky_replace(source, destination) -> None:
        nonlocal failures
        if Path(destination).name == "artifacts.json" and failures < 2:
            failures += 1
            raise PermissionError("simulated OneDrive sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(science_common.os, "replace", flaky_replace)
    store = ArtifactStore(tmp_path / "run", "TARGET-TEST")
    artifact = store.save_npz("example", "example_data", values=np.arange(3))

    assert failures == 2
    assert artifact.sha256 == sha256_file(tmp_path / "run" / str(artifact.path))
    assert store.manifest_path.is_file()
    assert not list(store.artifacts_directory.glob(".*.tmp"))


def test_invalid_tool_parameter_is_structured(repository_root: Path, tmp_path: Path) -> None:
    toolbox = ScienceToolbox(repository_root / "data" / "tess", tmp_path / "runs")
    result = toolbox.execute(
        ToolRequest(
            experiment_type=ExperimentType.NORMALIZATION,
            requested_by="pytest",
            parameters={"invented_knob": 42},
        ),
        _state("TARGET-X17"),
    )
    assert isinstance(result, ScientificFailure)
    assert result.status is ScientificStatus.INVALID_REQUEST
    assert result.reason_code == "UNKNOWN_PARAMETER"
