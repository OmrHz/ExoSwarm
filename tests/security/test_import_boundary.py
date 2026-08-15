from __future__ import annotations

from pathlib import Path

from exoswarm.security import catalog_import_violations


def test_agent_and_science_packages_cannot_import_security_catalog() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    assert catalog_import_violations(repository_root / "src") == []


def test_boundary_scanner_detects_direct_bypass(tmp_path) -> None:
    bad_module = tmp_path / "exoswarm" / "agents" / "bad.py"
    bad_module.parent.mkdir(parents=True)
    bad_module.write_text("from exoswarm.security import GroundTruthGate\n", encoding="utf-8")
    violations = catalog_import_violations(tmp_path)
    assert len(violations) == 1
    assert "GroundTruthGate" not in violations[0]  # report dependency, not source content
    assert "exoswarm.security" in violations[0]
