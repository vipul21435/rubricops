"""RubricOps: expert review and rubric-grading operations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rubricops")
except PackageNotFoundError:  # pragma: no cover - only hit when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
