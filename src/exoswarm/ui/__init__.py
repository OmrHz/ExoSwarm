"""Mission-control presentation layer for persisted ExoSwarm investigations.

The UI is intentionally read-only with respect to scientific state.  It consumes
validated run artifacts produced by the deterministic runtime and never opens the
private target catalog or source FITS headers.
"""

from .artifacts import MissionControlRun, RunPhase, load_run

__all__ = ["MissionControlRun", "RunPhase", "load_run"]
