"""Audio file output: 16-bit mono WAV, and AAC ``.m4a`` via ``afconvert``.

``afconvert`` is the macOS system encoder at ``/usr/bin/afconvert`` — the same
AAC path AVFoundation gives the Swift stack, so the two stacks' ``.m4a`` files
are comparable artifacts rather than two different codecs' opinions.

The WAV is the canonical deliverable: it is what the listening set and any
objective metric should read. The ``.m4a`` exists because that is what the
podcast pipeline ships.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Final

import numpy as np

#: macOS system encoder. Not a dependency we install; not available off macOS.
AFCONVERT: Final = "/usr/bin/afconvert"

#: AAC-LC target bitrate for the episode ``.m4a``.
#:
#: 64 kbps, not the podcast-typical 128: AAC-LC at 24 kHz mono (the qwen3 and
#: chatterbox output rate) *rejects* anything above 64 kbps — afconvert fails
#: with ``Couldn't set audio converter property ('!dat')`` — and 64 kbps is
#: already well past transparent for a 24 kHz mono speech signal. A rate the
#: encoder refuses falls back to afconvert's own choice rather than failing the
#: run; see :func:`encode_m4a`.
M4A_BITRATE: Final = 64_000

#: WAV subtype. 16-bit mono matches Produciesta's ``AdapterFormat.v1``.
WAV_SUBTYPE: Final = "PCM_16"


class EncodeError(RuntimeError):
    """Writing or converting an audio file failed."""


def write_wav(
    audio: np.ndarray,
    sample_rate: int,
    path: str | Path,
    *,
    subtype: str = WAV_SUBTYPE,
) -> Path:
    """Write a 1-D float waveform as a WAV file and return its path."""
    import soundfile as sf

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, np.asarray(audio, dtype=np.float32), int(sample_rate), subtype=subtype)
    return target


def afconvert_available(binary: str = AFCONVERT) -> bool:
    """True when the system ``afconvert`` can be invoked."""
    return Path(binary).is_file() or shutil.which(binary) is not None


def encode_m4a(
    wav_path: str | Path,
    m4a_path: str | Path,
    *,
    bitrate: int = M4A_BITRATE,
    binary: str = AFCONVERT,
) -> Path:
    """Convert a WAV to AAC in an MPEG-4 container via ``afconvert``.

    Raises :class:`EncodeError` with ``afconvert``'s own stderr attached, since
    its failures (unsupported sample rate, bad bitrate for the rate) are
    specific and worth surfacing verbatim.
    """
    source = Path(wav_path).expanduser()
    target = Path(m4a_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not afconvert_available(binary):
        raise EncodeError(
            f"{binary} not found; .m4a export needs the macOS system encoder "
            "(re-run with --no-m4a to write only the WAV)"
        )

    # AAC-LC, maximum codec quality. The bitrate is dropped on a retry because
    # the legal range depends on the sample rate and channel count, and a
    # rejected rate must not cost us the episode.
    base = [binary, "-f", "m4af", "-d", "aac", "-q", "127"]
    attempts = [
        base + ["-b", str(int(bitrate)), str(source), str(target)],
        base + [str(source), str(target)],
    ]

    failures: list[str] = []
    for command in attempts:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and target.is_file():
            return target
        failures.append(
            f"exit {completed.returncode}: "
            + (completed.stderr.strip() or completed.stdout.strip() or "no output")
        )

    raise EncodeError(
        f"afconvert failed converting {source} -> {target}; "
        + " | ".join(failures)
    )
