"""Static dependency checks that support the blindness protocol in CI."""

from __future__ import annotations

import ast
from pathlib import Path


class ImportBoundaryViolation(RuntimeError):
    pass


FORBIDDEN_PREFIXES = (
    "exoswarm.security",
    "exoswarm.catalog",
    "exoswarm.ground_truth",
)


def catalog_import_violations(source_root: str | Path) -> list[str]:
    """Find agent/science modules that directly import gated catalog capabilities.

    Runtime composition code may import security services.  Scientific functions
    and agent policy code may not: they receive identity-free artifacts/state via
    dependency injection.
    """

    root = Path(source_root).resolve()
    violations: list[str] = []
    for package_name in ("agents", "science"):
        package = root / "exoswarm" / package_name
        if not package.exists():
            continue
        for path in sorted(package.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                violations.append(f"{path}: syntax error prevented boundary audit: {exc}")
                continue
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                    # Any upward relative import reaching security is also blocked.
                    if node.level and (node.module or "").startswith("security"):
                        imported.append("exoswarm.security")
                for module in imported:
                    if module.startswith(FORBIDDEN_PREFIXES):
                        relative = path.relative_to(root)
                        violations.append(
                            f"{relative}:{getattr(node, 'lineno', '?')} imports {module}"
                        )
    return violations


def assert_catalog_import_boundary(source_root: str | Path) -> None:
    violations = catalog_import_violations(source_root)
    if violations:
        formatted = "\n".join(f"- {item}" for item in violations)
        raise ImportBoundaryViolation(
            f"agent/science packages may not import catalog/security capabilities:\n{formatted}"
        )
