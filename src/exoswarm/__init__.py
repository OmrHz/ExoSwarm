"""ExoSwarm: evidence-led, blinded investigation of TESS observations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exoswarm")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.0"

__all__ = ["__version__"]
