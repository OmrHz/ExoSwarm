"""Backend-only loader for the curated opaque target vault.

This module is in the deterministic runtime boundary, never imported by agent or
science packages. Identity-bearing catalog data and identity-free science manifests
remain physically separate on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exoswarm.domain.models import (
    ArtifactRef,
    CatalogMeasurement,
    GroundTruthRecord,
    ProvenanceRecord,
)
from exoswarm.science.common import MAST_TESS_COLLECTION_URL, load_science_manifest
from exoswarm.security.blindness import OpaqueTargetVault


class PrivateTargetManifestError(ValueError):
    pass


def load_demo_vault(data_directory: str | Path) -> tuple[OpaqueTargetVault, tuple[str, ...]]:
    """Load the private mapping once into a capability-gated vault."""

    root = Path(data_directory).resolve()
    private_path = (root / "private" / "targets.json").resolve()
    if root not in private_path.parents:
        raise PrivateTargetManifestError("private manifest escaped data directory")
    try:
        payload = json.loads(private_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PrivateTargetManifestError(
            f"missing private target manifest: {private_path}"
        ) from exc
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise PrivateTargetManifestError("private target manifest has no targets")

    vault = OpaqueTargetVault()
    opaque_ids: list[str] = []
    for raw in targets:
        if not isinstance(raw, dict):
            raise PrivateTargetManifestError("target record must be an object")
        opaque = _required_string(raw, "opaque_target_id")
        identity = _required_string(raw, "actual_target_identity")
        science = load_science_manifest(root / "tess", opaque)
        target_root = (root / "tess" / opaque).resolve()
        artifacts = [
            ArtifactRef(
                artifact_id=f"{opaque}:cached-{role}",
                path=str((target_root / product.path).resolve()),
                sha256=product.sha256,
                media_type=product.media_type,
                role=role,
                source_uri=MAST_TESS_COLLECTION_URL,
            )
            for role, product in science.products.items()
        ]
        ground_truth_raw = raw.get("ground_truth")
        if not isinstance(ground_truth_raw, dict):
            raise PrivateTargetManifestError(f"{opaque} has no ground_truth object")
        measurements_raw = ground_truth_raw.get("measurements", {})
        if not isinstance(measurements_raw, dict):
            raise PrivateTargetManifestError(f"{opaque} ground_truth.measurements is invalid")
        measurements = {
            str(name): CatalogMeasurement.model_validate(value)
            for name, value in measurements_raw.items()
        }
        catalog_name = _required_string(ground_truth_raw, "catalog_name")
        catalog_status = _required_string(ground_truth_raw, "catalog_status")
        source_uri = _required_string(ground_truth_raw, "source_uri")
        ground_truth = GroundTruthRecord(
            actual_target_identity=identity,
            catalog_name=catalog_name,
            catalog_status=catalog_status,
            measurements=measurements,
            provenance=[
                ProvenanceRecord(
                    source=catalog_name,
                    source_uri=source_uri,
                    notes=[
                        "Loaded by the backend-private target registry.",
                        "This record becomes callable only after a verified result lock.",
                    ],
                )
            ],
        )
        vault.register(
            opaque_target_id=opaque,
            real_target_identity=identity,
            artifacts=artifacts,
            ground_truth=ground_truth,
        )
        opaque_ids.append(opaque)
    if len(set(opaque_ids)) != len(opaque_ids):
        raise PrivateTargetManifestError("duplicate opaque target IDs")
    return vault, tuple(opaque_ids)


def blind_target_summaries(data_directory: str | Path) -> list[dict[str, Any]]:
    """Return identity-safe target choices for the CLI and pre-reveal UI."""

    root = Path(data_directory).resolve()
    tess_root = root / "tess"
    summaries: list[dict[str, Any]] = []
    for path in sorted(tess_root.glob("TARGET-*/science_manifest.json")):
        opaque = path.parent.name
        manifest = load_science_manifest(tess_root, opaque)
        summaries.append(
            {
                "opaque_target_id": opaque,
                "mission": manifest.mission,
                "sector": manifest.sector,
                "cadence_seconds": manifest.cadence_seconds,
                "available_product_roles": sorted(manifest.products),
            }
        )
    return summaries


def _required_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PrivateTargetManifestError(f"missing non-empty field {name!r}")
    return value


__all__ = ["PrivateTargetManifestError", "blind_target_summaries", "load_demo_vault"]
