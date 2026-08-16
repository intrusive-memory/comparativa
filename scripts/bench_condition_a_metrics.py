#!/usr/bin/env python3
"""Write ``bench/A/metrics.json`` from the condition-A runs produced by
``scripts/bench_condition_a.sh`` (OPERATION BATTLING BARDS, Sortie 8).

Reads, per episode directory under ``bench/A/``:

* ``run.json``     — command, exit code, timestamps, Produciesta identity
* ``time.txt``     — the child's stderr **and** the ``/usr/bin/time -l`` block
                     (wall-clock ``real`` seconds; ``maximum resident set
                     size`` in bytes on macOS)
* ``generate.log`` — the NDJSON progress stream (one event per Element)
* ``<ep>.m4a.vtt`` — Produciesta's sidecar transcript: one cue per Element,
                     which is where line count and voiced-audio seconds come
                     from (the Swift stack exposes no manifest)
* ``<ep>.m4a``     — decoded to wav with ``afconvert`` and measured with OUR
                     meter (``pyloudnorm``, the same code path the Python
                     conditions use) so the report can state a loudness delta.
* ``notes.txt``    — optional; one free-form caveat per line, appended to the
                     entry's ``notes``. This is how a human records something
                     the wrapper cannot see, e.g. "another process was
                     competing for the GPU during this run".

The decoded ``<ep>.wav`` is **kept**, not thrown away: ``eval.blind`` discovers
listening clips by globbing ``*.wav`` under each episode dir, so without it the
Swift baseline would silently drop out of Sortie 10's blinded listening set.

Two metrics files are written, which ``metrics.load_entries`` explicitly
supports (it de-duplicates on ``(stack, condition, episode)``): the aggregate
``bench/A/metrics.json`` the plan asks for, and a per-episode
``bench/A/<episode>/metrics.json``, which is the layout ``eval.blind`` and
``eval.metrics`` glob for.

Everything the Swift stack genuinely cannot report (model load time, per-line
truncation counts, the normalization shortfall it never applies) is written as
``null`` rather than guessed.

Usage::

    uv run python scripts/bench_condition_a_metrics.py [--bench-dir bench/A]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from comparativa.bench import make_entry, write_metrics  # noqa: E402
from comparativa.generation.assembly import (  # noqa: E402
    LOUDNESS_METER,
    TARGET_LUFS,
    measure_lufs,
)

CONDITION = "A"
STACK = "swift"
SPEECH_POLICY = "produciesta-parity"
LABEL = "Swift baseline — Produciesta/SwiftVoxAlta, Qwen3-TTS 1.7B bf16, .vox production voices"
VOICES = ".vox"

#: ``/usr/bin/time -l``: "      350.22 real        57.12 user       124.99 sys"
_TIME_REAL = re.compile(r"^\s*([0-9.]+)\s+real\s+", re.MULTILINE)
#: ``/usr/bin/time -l``: "          9241870336  maximum resident set size"
_TIME_MAXRSS = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)
_TIME_FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$", re.MULTILINE)
#: WebVTT timestamps: "00:01.280 --> 00:22.570" or "00:00:01.280 --> ..."
_VTT_CUE = re.compile(
    r"^((?:\d+:)?\d{2}:\d{2}\.\d{3})\s+-->\s+((?:\d+:)?\d{2}:\d{2}\.\d{3})", re.MULTILINE
)
#: The repo id SwiftAcervo resolves for the render model, as logged by the run.
_ACERVO_REPO = re.compile(r"component '(mlx-community[_/][A-Za-z0-9._-]+)'")


def _vtt_seconds(stamp: str) -> float:
    parts = stamp.split(":")
    seconds = float(parts[-1])
    if len(parts) >= 2:
        seconds += 60.0 * int(parts[-2])
    if len(parts) >= 3:
        seconds += 3600.0 * int(parts[-3])
    return seconds


def _decode_to_wav(m4a: Path, dest: Path) -> None:
    """Decode the Produciesta .m4a to 32-bit float wav via ``afconvert``."""
    subprocess.run(
        ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEF32", str(m4a), str(dest)],
        check=True,
        capture_output=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host() -> dict[str, Any]:
    host: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import os

        host["cpu_count"] = os.cpu_count()
        host["memory_bytes"] = int(
            subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except Exception:  # pragma: no cover - informational only
        pass
    return host


def _checkpoint(stderr_text: str) -> str | None:
    """The render checkpoint the run actually loaded.

    SwiftAcervo logs every registered component; the render model is the Base
    checkpoint (``VoxAltaConfig.default.renderModel``), and the run's own log
    confirms which family was loaded ("Loaded Qwen3-TTS model (base)").
    """
    names = {m.replace("_", "/", 1) for m in _ACERVO_REPO.findall(stderr_text)}
    base = sorted(n for n in names if "Base" in n)
    if base and "Loaded Qwen3-TTS model (base)" in stderr_text:
        return base[0]
    return base[0] if base else None


def _episode_entry(run_dir: Path) -> dict[str, Any]:
    episode = run_dir.name
    run_meta = json.loads((run_dir / "run.json").read_text())
    stderr_text = (run_dir / "time.txt").read_text(errors="replace")
    m4a = run_dir / f"{episode}.m4a"
    vtt = run_dir / f"{episode}.vtt"
    if not vtt.exists():
        vtt = run_dir / f"{episode}.m4a.vtt"

    real = _TIME_REAL.findall(stderr_text)
    maxrss = _TIME_MAXRSS.findall(stderr_text)
    footprint = _TIME_FOOTPRINT.findall(stderr_text)
    wall_seconds = float(real[-1]) if real else None
    peak_rss = int(maxrss[-1]) if maxrss else None

    returncode = int(run_meta.get("returncode", 1))
    status = "ok" if returncode == 0 and m4a.exists() else "failed"

    # --- lines and voiced audio, from the sidecar transcript -----------------
    cues: list[tuple[float, float]] = []
    if vtt.exists():
        cues = [(_vtt_seconds(a), _vtt_seconds(b)) for a, b in _VTT_CUE.findall(vtt.read_text())]
    audio_seconds = round(sum(end - start for start, end in cues), 3) if cues else None

    events = [
        json.loads(line)
        for line in (run_dir / "generate.log").read_text().splitlines()
        if line.startswith("{")
    ]
    script_line_count = max((e.get("total", 0) for e in events), default=0) or None
    failed_events = sum(1 for e in events if e.get("outcome") == "failed")

    # --- loudness, measured with OUR meter -----------------------------------
    duration_seconds = sample_rate = None
    mean_output_lufs = episode_lufs = None
    unmeasurable = 0
    wav = run_dir / f"{episode}.wav"
    if m4a.exists():
        _decode_to_wav(m4a, wav)  # kept: eval.blind discovers clips by *.wav
        audio, sample_rate = sf.read(str(wav), dtype="float64", always_2d=False)
        if audio.ndim > 1:  # measure the mono sum the listener hears
            audio = audio.mean(axis=1)
        duration_seconds = round(len(audio) / float(sample_rate), 3)
        episode_lufs = measure_lufs(np.asarray(audio), int(sample_rate))
        per_cue: list[float] = []
        for start, end in cues:
            lo = max(0, int(start * sample_rate))
            hi = min(len(audio), int(end * sample_rate))
            if hi <= lo:
                unmeasurable += 1
                continue
            value = measure_lufs(np.asarray(audio[lo:hi]), int(sample_rate))
            if value is None:
                unmeasurable += 1
            else:
                per_cue.append(value)
        if per_cue:
            mean_output_lufs = round(sum(per_cue) / len(per_cue), 3)

    rtf = (
        round(wall_seconds / audio_seconds, 4)
        if wall_seconds and audio_seconds
        else None
    )
    gap_seconds = (
        round(duration_seconds - audio_seconds, 3)
        if duration_seconds is not None and audio_seconds is not None
        else None
    )

    extra_notes: list[str] = []
    notes_file = run_dir / "notes.txt"
    if notes_file.exists():
        extra_notes = [
            line.strip() for line in notes_file.read_text().splitlines() if line.strip()
        ]

    screenplay = Path(run_meta["episode_path"])
    return make_entry(
        condition=CONDITION,
        stack=STACK,
        label=LABEL,
        engine="produciesta",
        checkpoint=_checkpoint(stderr_text),
        voices=VOICES,
        episode=episode,
        episode_path=str(screenplay),
        episode_sha256=_sha256(screenplay) if screenplay.exists() else None,
        speech_policy=SPEECH_POLICY,
        status=status,
        error=None if status == "ok" else f"produciesta exited {returncode}",
        wall_seconds=wall_seconds,
        real_time_factor=rtf,
        model_load_seconds=None,  # the Swift CLI does not report load time
        peak_rss_bytes=peak_rss,
        performance_extra={
            "peak_memory_footprint_bytes": int(footprint[-1]) if footprint else None,
            "generate_seconds": None,
            "process_wall_seconds": wall_seconds,
        },
        audio_seconds=audio_seconds,
        audio_extra={
            "duration_seconds": duration_seconds,
            "gap_seconds": gap_seconds,
            "sample_rate": int(sample_rate) if sample_rate else None,
            # `loudness` has a fixed key set in the frozen schema, and
            # `*_extra` is the sanctioned extension point, so these three ride
            # here rather than being silently dropped.
            "episode_integrated_lufs": (
                round(episode_lufs, 3) if episode_lufs is not None else None
            ),
            "loudness_reference_target_lufs": TARGET_LUFS,
            "loudness_unmeasurable_lines": unmeasurable,
        },
        line_count=len(cues) or None,
        lines_extra={
            "placed_line_count": len(cues) or None,
            "script_line_count": script_line_count,
            "truncated_lines": None,  # not exposed by the Swift stack
            "truncation_retry_lines": None,
            "overrun_lines": None,
            "failed_elements": failed_events,
        },
        loudness={
            "meter": LOUDNESS_METER,
            "target_lufs": None,  # Produciesta does not normalize to a LUFS target
            "mean_output_lufs": mean_output_lufs,
            "mean_shortfall_db": None,
            "max_shortfall_db": None,
            "unnormalized_lines": len(cues) or None,
        },
        tool_versions={
            "produciesta": run_meta.get("produciesta_version"),
            "swift_voxalta": run_meta.get("swift_voxalta_version"),
            "produciesta_spec_hash": run_meta.get("spec_hash"),
            "macos": run_meta.get("macos"),
        },
        host=_host(),
        run={
            "started_at": run_meta.get("started_at"),
            "finished_at": run_meta.get("finished_at"),
            "command": run_meta.get("command"),
            "returncode": returncode,
            "log": str(run_dir / "time.txt"),
        },
        outputs={
            # Produciesta exports .m4a only; the .wav is our afconvert decode of
            # it, kept so the blinded listening set can include this condition.
            "wav": str(wav) if wav.exists() else None,
            "m4a": str(m4a) if m4a.exists() else None,
            "manifest": None,  # no manifest on the Swift stack
            "vtt": str(vtt) if vtt.exists() else None,
        },
        notes=[
            "peak RSS from /usr/bin/time -l (maximum resident set size, bytes)",
            "wall-clock is the same /usr/bin/time -l run; it includes model load",
            "model_load_seconds unavailable: the produciesta CLI does not report it",
            "line_count/audio_seconds derived from Produciesta's sidecar .vtt "
            "(one cue per Element); the Swift stack writes no manifest",
            "loudness measured with comparativa's own pyloudnorm meter after "
            "afconvert decode; Produciesta applies no LUFS target, so the "
            "shortfall fields are null rather than zero",
            "outputs.wav is an afconvert decode of the delivered .m4a, not a "
            "native render: the AAC round-trip is in the listened signal",
            "audio regenerated fresh for this benchmark; the granville "
            "audio/*.m4a of 2026-08-09 are reference-only and untouched",
            *extra_notes,
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default=str(REPO_ROOT / "bench" / "A"))
    args = parser.parse_args(argv)

    bench_dir = Path(args.bench_dir).resolve()
    run_dirs = sorted(d for d in bench_dir.iterdir() if d.is_dir() and (d / "run.json").exists())
    if not run_dirs:
        print(f"error: no condition-A runs under {bench_dir}", file=sys.stderr)
        return 2

    entries = [_episode_entry(d) for d in run_dirs]
    out = bench_dir / "metrics.json"
    write_metrics(out, entries, generated_by="sortie-8 condition A (produciesta)")
    # Per-episode copies: the layout eval.blind / eval.metrics glob for.
    for run_dir, entry in zip(run_dirs, entries):
        write_metrics(
            run_dir / "metrics.json",
            [entry],
            generated_by="sortie-8 condition A (produciesta)",
        )

    for entry in entries:
        perf, audio, loud = entry["performance"], entry["audio"], entry["loudness"]
        print(
            f"{entry['episode']}: status={entry['status']} "
            f"lines={entry['lines']['line_count']} "
            f"audio={audio['audio_seconds']}s wall={perf['wall_seconds']}s "
            f"rtf={perf['real_time_factor']} "
            f"peakRSS={(perf['peak_rss_bytes'] or 0) / 2**30:.2f} GiB "
            f"meanLUFS={loud['mean_output_lufs']} "
            f"episodeLUFS={audio['episode_integrated_lufs']} "
            f"ckpt={entry['checkpoint']}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
