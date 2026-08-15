"""Performance capture for one condition run (FR-12).

Three of the four numbers FR-12 asks for — wall-clock, real-time factor, and
model load time — are already measured *inside* generation and land in the
manifest's ``totals`` (``wall_seconds``, ``real_time_factor``, ``load_seconds``).
The bench layer reads them from there rather than re-deriving them, so the
report and the manifest can never disagree.

**Peak RSS is the one number generation does not know**, and it is the reason
each condition runs in its own subprocess: a 1.7 B checkpoint loaded for
condition C stays resident in the interpreter, so an in-process condition E
would inherit C's footprint and report a peak RSS that is mostly someone else's.
One process per (condition, episode) makes the number mean what it says.

:class:`PeakRSS` polls ``psutil`` at :data:`SAMPLE_INTERVAL` and keeps the
maximum of the process tree's resident set. Polling can only *under*-report a
spike shorter than the interval; at 50 ms against a multi-minute model load
that is not a practical concern, and the alternative (``ru_maxrss`` of
``RUSAGE_CHILDREN``) is a high-water mark across *all* reaped children and so
cannot be attributed to a single run at all.
"""

from __future__ import annotations

import platform
import sys
import threading
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

import psutil

from .. import MLX_AUDIO_VERSION, __version__

#: How often the sampler thread reads the process tree's RSS.
SAMPLE_INTERVAL: Final = 0.05

#: Distributions reported in every Python-stack metrics entry.
TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "mlx-audio",
    "mlx",
    "psutil",
    "numpy",
    "pyloudnorm",
    "soundfile",
    "jouvence",
)


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - all are pinned deps
        return None


def python_tool_versions() -> dict[str, str | None]:
    """Tool versions for a Python-stack metrics entry.

    The Swift side (Sortie 8) supplies its own dict — a ``produciesta`` version
    string and whatever else it can determine. Nothing in the schema requires
    these particular keys.
    """
    versions: dict[str, str | None] = {
        "comparativa": __version__,
        "python": platform.python_version(),
        "mlx_audio_pinned": MLX_AUDIO_VERSION,
    }
    for name in TRACKED_PACKAGES:
        versions[name.replace("-", "_")] = _distribution_version(name)
    return versions


def host_info() -> dict[str, Any]:
    """The machine the run happened on, for cross-run comparability."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


@dataclass
class RSSSample:
    """The result of one sampling session."""

    peak_rss_bytes: int = 0
    first_rss_bytes: int = 0
    samples: int = 0

    @property
    def ok(self) -> bool:
        return self.samples > 0


class PeakRSS:
    """Sample a process tree's resident set size until it exits.

    Use as a context manager around a running :class:`subprocess.Popen`::

        with PeakRSS(child.pid) as rss:
            child.wait()
        rss.result.peak_rss_bytes
    """

    def __init__(self, pid: int, *, interval: float = SAMPLE_INTERVAL) -> None:
        self.pid = pid
        self.interval = interval
        self.result = RSSSample()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- sampling -----------------------------------------------------------

    def _tree_rss(self, process: psutil.Process) -> int:
        total = 0
        try:
            with process.oneshot():
                total += process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0
        return total

    def _run(self) -> None:
        try:
            process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:  # pragma: no cover - race on a fast exit
            return
        while not self._stop.is_set():
            rss = self._tree_rss(process)
            if rss:
                if self.result.samples == 0:
                    self.result.first_rss_bytes = rss
                self.result.samples += 1
                self.result.peak_rss_bytes = max(self.result.peak_rss_bytes, rss)
            elif self.result.samples:
                break  # the tree is gone; nothing left to sample
            if self._stop.wait(self.interval):
                return

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "PeakRSS":
        self._thread = threading.Thread(
            target=self._run, name=f"peak-rss-{self.pid}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> RSSSample:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.result

    def __enter__(self) -> "PeakRSS":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def self_peak_rss_bytes() -> int:
    """Current process RSS — the floor for any in-process measurement."""
    return int(psutil.Process().memory_info().rss)


__all__ = [
    "SAMPLE_INTERVAL",
    "TRACKED_PACKAGES",
    "PeakRSS",
    "RSSSample",
    "host_info",
    "python_tool_versions",
    "self_peak_rss_bytes",
]
