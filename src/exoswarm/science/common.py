"""Shared deterministic-science data and artifact utilities.

Nothing in this package imports the ground-truth/security layer.  Target manifests
contain only an opaque identifier and the minimum geometry needed to execute the
pixel diagnostic; real identities and catalog dispositions live behind the vault.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from exoswarm.domain.models import ArtifactRef, ProvenanceRecord

TOOL_VERSION = "0.1.0"
MAST_TESS_COLLECTION_URL = "https://archive.stsci.edu/missions-and-data/tess/data-products"
DETERMINISTIC_STACK_VERSIONS = {
    "exoswarm-science": TOOL_VERSION,
    "astropy": version("astropy"),
    "numpy": version("numpy"),
    "scipy": version("scipy"),
    "matplotlib": version("matplotlib"),
}


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CachedProduct(_ManifestModel):
    path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: str = "application/fits"

    @model_validator(mode="after")
    def relative_safe_path(self) -> CachedProduct:
        candidate = Path(self.path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("cached product paths must be safe and relative")
        return self


class NeighborContext(_ManifestModel):
    catalog: str
    queried_at: str
    nearest_neighbor_separation_arcsec: float = Field(gt=0)
    nearest_neighbor_delta_tmag: float
    nearest_neighbor_pixel_offset_x: float
    nearest_neighbor_pixel_offset_y: float
    search_radius_arcsec: float = Field(gt=0)


class ScienceManifest(_ManifestModel):
    schema_version: str
    opaque_target_id: str = Field(pattern=r"^TARGET-[A-Z0-9][A-Z0-9-]*$")
    mission: Literal["TESS"]
    pipeline: Literal["SPOC"]
    sector: int = Field(gt=0)
    cadence_seconds: float = Field(gt=0)
    retrieved_at: str
    archive_collection_url: str
    products: dict[Literal["light_curve", "target_pixel"], CachedProduct]
    neighbor_context: NeighborContext | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_products_exist(self) -> ScienceManifest:
        if set(self.products) != {"light_curve", "target_pixel"}:
            raise ValueError("science manifest requires light_curve and target_pixel products")
        return self


@dataclass(frozen=True, slots=True)
class LightCurveData:
    """In-process array container; never serialized into an agent context."""

    time_btjd: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray
    sector: int
    cadence_seconds: float
    crowdsap: float | None
    flfrcsap: float | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_science_manifest(data_root: str | Path, opaque_target_id: str) -> ScienceManifest:
    root = Path(data_root).resolve()
    path = root / opaque_target_id / "science_manifest.json"
    try:
        manifest = ScienceManifest.model_validate_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"no cached science manifest for {opaque_target_id}") from exc
    if manifest.opaque_target_id != opaque_target_id:
        raise ValueError("manifest opaque target does not match requested target")
    return manifest


def product_path(
    data_root: str | Path, manifest: ScienceManifest, role: Literal["light_curve", "target_pixel"]
) -> Path:
    target_root = (Path(data_root).resolve() / manifest.opaque_target_id).resolve()
    path = (target_root / manifest.products[role].path).resolve()
    if target_root not in path.parents:
        raise ValueError("cached product escaped opaque target directory")
    return path


def verify_cached_product(path: Path, product: CachedProduct) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != product.size_bytes:
        raise ValueError(f"cached product size mismatch: {path.name}")
    if sha256_file(path) != product.sha256:
        raise ValueError(f"cached product SHA-256 mismatch: {path.name}")


def source_artifact(manifest: ScienceManifest, role: str) -> ArtifactRef:
    product = manifest.products[role]  # type: ignore[index]
    # Deliberately omit the local path and exact product URL: both contain backend
    # identifiers.  The opaque manifest and private vault retain full provenance.
    return ArtifactRef(
        artifact_id=f"{manifest.opaque_target_id}:cached-{role}",
        sha256=product.sha256,
        media_type=product.media_type,
        role=f"cached_{role}",
        source_uri=MAST_TESS_COLLECTION_URL,
    )


def tess_provenance(manifest: ScienceManifest, role: str) -> ProvenanceRecord:
    product = manifest.products[role]  # type: ignore[index]
    return ProvenanceRecord(
        source="NASA MAST / TESS SPOC cached mission product",
        source_uri=MAST_TESS_COLLECTION_URL,
        source_sha256=product.sha256,
        software=dict(DETERMINISTIC_STACK_VERSIONS),
        notes=[
            f"sector={manifest.sector}",
            f"nominal_cadence_seconds={manifest.cadence_seconds:g}",
            "Exact TIC-bearing product URI is retained only in the backend-private manifest.",
        ],
    )


class ArtifactStore:
    """Write deterministic run-local science artifacts and a UI-safe index."""

    def __init__(self, run_directory: str | Path, opaque_target_id: str) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.opaque_target_id = opaque_target_id
        self.artifacts_directory = self.run_directory / "artifacts"
        self.science_directory = self.artifacts_directory / "science"
        self.science_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.artifacts_directory / "artifacts.json"

    def save_npz(self, stem: str, role: str, **arrays: np.ndarray) -> ArtifactRef:
        path = self.science_directory / f"{stem}.npz"
        _write_deterministic_npz(path, arrays)
        return self._register(path, role=role, media_type="application/x-npz")

    def save_figure(self, stem: str, role: str, figure: object) -> ArtifactRef:
        path = self.science_directory / f"{stem}.png"
        # Matplotlib's PNG output is byte-stable for fixed data/version when volatile
        # metadata is explicitly suppressed.
        figure.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
            facecolor="#08111f",
            metadata={"Software": f"ExoSwarm science {TOOL_VERSION}"},
        )
        return self._register(path, role=role, media_type="image/png")

    def existing(self, stem: str, role: str, media_type: str = "application/x-npz") -> ArtifactRef:
        suffix = ".npz" if media_type == "application/x-npz" else ".png"
        path = self.science_directory / f"{stem}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(path)
        return self._register(path, role=role, media_type=media_type)

    def path(self, stem: str, suffix: str = ".npz") -> Path:
        return self.science_directory / f"{stem}{suffix}"

    def _register(self, path: Path, *, role: str, media_type: str) -> ArtifactRef:
        relative = path.relative_to(self.run_directory).as_posix()
        artifact = ArtifactRef(
            artifact_id=f"{self.opaque_target_id}:{role}:{path.suffix.lstrip('.')}",
            path=relative,
            sha256=sha256_file(path),
            media_type=media_type,
            role=role,
        )
        entries: dict[str, dict[str, str]] = {}
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            entries = {item["artifact_id"]: item for item in raw.get("artifacts", [])}
        entries[artifact.artifact_id] = {
            "artifact_id": artifact.artifact_id,
            "relative_path": relative,
            "sha256": artifact.sha256 or "",
            "media_type": media_type,
            "role": role,
        }
        payload = {
            "schema_version": "1.0",
            "opaque_target_id": self.opaque_target_id,
            "path_base": "run_directory",
            "artifacts": [entries[key] for key in sorted(entries)],
        }
        _atomic_write_bytes(
            self.manifest_path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return artifact


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write NPZ with fixed ZIP timestamps and sorted members for stable hashes."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or "\\" in name:
                    raise ValueError(f"unsafe NPZ member name: {name!r}")
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a small artifact, tolerating brief Windows sync locks."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _replace_with_retry(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _replace_with_retry(source: Path, destination: Path, *, attempts: int = 8) -> None:
    """Retry only transient sharing violations; surface every other filesystem error."""

    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.025 * (2**attempt), 0.4))


def robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    return float(1.4826 * mad)


def setup_science_axes(axis: object, *, xlabel: str, ylabel: str) -> None:
    axis.set_facecolor("#0d1727")
    axis.tick_params(colors="#a9bdd6")
    axis.xaxis.label.set_color("#c5d5e8")
    axis.yaxis.label.set_color("#c5d5e8")
    axis.title.set_color("#e8f2ff")
    for spine in axis.spines.values():
        spine.set_color("#29415f")
    axis.grid(color="#29415f", alpha=0.22, linewidth=0.6)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
