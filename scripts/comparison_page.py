#!/usr/bin/env python3
"""Write ``bench/comparison.html`` — every condition's generation, side by side.

Reads the completed cells under ``bench/<condition>/<episode>/`` and emits one
self-contained local HTML page with an audio player per condition per episode,
plus a clickable transcript that seeks **every** player to the same script
line (each condition lands on its own timeline offset for that line).

Line offsets come from:

* Python conditions (B/C/D/E/G/T) — ``manifest.json`` ``lines[].offset_seconds``.
* Condition A (Swift) — the Produciesta ``.vtt`` sidecar. Cue counts can differ
  from the parity line count (Produciesta wraps intro/outro on some exports),
  so cues are aligned to script lines by normalized text with a two-pointer
  sweep; unmatched lines simply do not seek A.

The page references the audio by *relative* path, so it must stay in ``bench/``
(which is gitignored — the page is a regenerable artifact, like the audio):

    uv run python scripts/comparison_page.py && open bench/comparison.html
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "bench"
OUT = BENCH / "comparison.html"

#: Display order and blurbs. Conditions absent from bench/ are listed as such.
CONDITION_LABELS: dict[str, str] = {
    "A": "Swift · Produciesta · production .vox voices",
    "B": "Python · qwen3-1.7b Base · .vox-cloned (ICL)",
    "C": "Python · qwen3-1.7b · preset speakers",
    "D": "Python · chatterbox · default voice",
    "E": "Python · Soprano-80M · single voice",
    "F": "Swift · mlx-audio-swift Soprano",
    "G": "Python · chatterbox · .vox-cloned",
    "T": "Python · chatterbox-turbo · default voice",
}

#: Default/single-voice renders whose player is redundant once the same
#: engine's **cloned** render exists (the stats table still lists them).
#: E (Soprano) has no entry: Soprano cannot clone — its one voice is baked in
#: (soprano.py:387 discards the `voice` argument) — so E is Soprano's only
#: render and always keeps its player.
SUPERSEDED_BY_CLONE: dict[str, str] = {
    "D": "G",  # chatterbox default -> chatterbox .vox-cloned
}

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


@dataclass
class Row:
    condition: str
    src: str
    duration: float | None
    rtf: float | None
    wall: float | None
    line_count: int | None
    offsets: list[float | None]


_CUE_ID = re.compile(r"^[0-9A-Fa-f-]{8,}$")


def parse_vtt(path: Path) -> list[tuple[float, str]]:
    """``(start_seconds, normalized_text)`` per cue, in order.

    Produciesta writes cues **contiguously** (a UUID id line straight after the
    previous cue's text, no blank separators), so a pending cue is flushed on
    each new timestamp line rather than on blank lines.
    """
    cues: list[tuple[float, str]] = []
    start: float | None = None
    text: list[str] = []

    def flush() -> None:
        nonlocal start, text
        if start is not None:
            cues.append((start, _norm(" ".join(text))))
        start, text = None, []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "-->" in line:
            flush()
            m = _TIME.match(line)
            if m:
                h, mi, s, ms = (int(g) if g else 0 for g in m.groups())
                start = h * 3600 + mi * 60 + s + ms / 1000.0
        elif line and start is not None and not _CUE_ID.match(line):
            text.append(_TAG.sub("", line))
    flush()
    return cues


def align_vtt(cues: list[tuple[float, str]], lines: list[dict[str, Any]]) -> list[float | None]:
    """Two-pointer text alignment of VTT cues to script lines."""
    offsets: list[float | None] = []
    cursor = 0
    for line in lines:
        wanted = _norm(str(line.get("text", "")))
        found: float | None = None
        for probe in range(cursor, len(cues)):
            if cues[probe][1] == wanted:
                found = cues[probe][0]
                cursor = probe + 1
                break
        offsets.append(found)
    return offsets


def load_metrics(cell: Path) -> dict[str, Any]:
    try:
        doc = json.loads((cell / "metrics.json").read_text(encoding="utf-8"))
        return (doc.get("entries") or [{}])[0]
    except (OSError, json.JSONDecodeError):
        return {}


def collect() -> tuple[dict[str, dict[str, Row]], list[dict[str, Any]] | None, dict[str, list[dict[str, Any]]], list[str]]:
    """Scan bench/ into per-episode rows plus the shared transcript."""
    episodes: dict[str, dict[str, Row]] = {}
    transcripts: dict[str, list[dict[str, Any]]] = {}
    skipped: list[str] = []

    for cond in sorted(CONDITION_LABELS):
        cdir = BENCH / cond
        if not cdir.is_dir():
            continue
        if (cdir / "SKIPPED.md").is_file():
            skipped.append(cond)
            continue
        for cell in sorted(p for p in cdir.iterdir() if p.is_dir()):
            episode = cell.name
            audio = next(iter(sorted(cell.glob("*.m4a")) or sorted(cell.glob("*.wav"))), None)
            if audio is None:
                continue
            entry = load_metrics(cell)
            perf = entry.get("performance") or {}
            audio_info = entry.get("audio") or {}
            line_info = entry.get("lines") or {}

            manifest_lines: list[dict[str, Any]] | None = None
            manifest_path = cell / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_lines = manifest.get("lines") or []
                transcripts.setdefault(episode, [
                    {"character": ln.get("character", ""), "text": ln.get("text", "")}
                    for ln in manifest_lines
                ])

            offsets: list[float | None] = []
            if manifest_lines is not None:
                offsets = [ln.get("offset_seconds") for ln in manifest_lines]
            episodes.setdefault(episode, {})[cond] = Row(
                condition=cond,
                src=f"{cond}/{episode}/{audio.name}",
                duration=audio_info.get("audio_seconds"),
                rtf=perf.get("real_time_factor"),
                wall=perf.get("wall_seconds"),
                line_count=line_info.get("line_count"),
                offsets=offsets,
            )

    # Align condition A's VTT cues to the shared transcript, per episode.
    for episode, rows in episodes.items():
        row = rows.get("A")
        transcript = transcripts.get(episode)
        if row is None or transcript is None:
            continue
        vtt = next(iter((BENCH / "A" / episode).glob("*.vtt")), None)
        if vtt is not None:
            row.offsets = align_vtt(parse_vtt(vtt), transcript)

    return episodes, None, transcripts, skipped


STYLE = """
:root { color-scheme: light dark;
  --bg:#f7f7f5; --panel:#ffffff; --ink:#1a1a1a; --muted:#6b6b66; --line:#e3e3de;
  --accent:#7c5cff; --hi:#efeaff; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#131316; --panel:#1c1c21; --ink:#ececea; --muted:#9a9a94; --line:#2a2a30;
  --accent:#9d85ff; --hi:#2b2440; } }
* { box-sizing:border-box }
body { margin:0; padding:2rem clamp(1rem,4vw,3rem); background:var(--bg); color:var(--ink);
  font:15px/1.5 -apple-system, "Helvetica Neue", sans-serif; }
h1 { font-size:1.5rem; margin:0 0 .25rem } h2 { font-size:1.15rem; margin:2.5rem 0 .75rem }
.sub { color:var(--muted); margin:0 0 1.5rem }
table.perf { border-collapse:collapse; width:100%; max-width:64rem; font-size:.85rem; margin:0 0 1rem }
table.perf th, table.perf td { text-align:left; padding:.3rem .7rem; border-bottom:1px solid var(--line) }
table.perf th { color:var(--muted); font-weight:600 }
table.perf td.num, table.perf th.num { text-align:right; font-variant-numeric:tabular-nums }
.episode { display:grid; grid-template-columns: minmax(24rem, 3fr) minmax(18rem, 2fr); gap:1.25rem; align-items:start }
@media (max-width: 900px){ .episode { grid-template-columns: 1fr } }
.players { display:flex; flex-direction:column; gap:.6rem }
.row { display:grid; grid-template-columns: 2.2rem 1fr; gap:.75rem; align-items:center;
  background:var(--panel); border:1px solid var(--line); border-radius:.65rem; padding:.6rem .8rem }
.badge { width:2.2rem; height:2.2rem; border-radius:.5rem; background:var(--hi); color:var(--accent);
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.05rem }
.row .meta { display:flex; justify-content:space-between; gap:1rem; font-size:.82rem; color:var(--muted); flex-wrap:wrap }
.row .meta strong { color:var(--ink); font-weight:600 }
.row audio { width:100%; height:2rem; margin-top:.35rem }
.controls { display:flex; gap:.5rem; margin:.75rem 0; flex-wrap:wrap; align-items:center }
.controls button { background:var(--panel); border:1px solid var(--line); color:var(--ink);
  border-radius:.5rem; padding:.35rem .8rem; font:inherit; font-size:.85rem; cursor:pointer }
.controls button:hover { border-color:var(--accent) }
.controls label { font-size:.85rem; color:var(--muted); display:flex; gap:.35rem; align-items:center }
.transcript { background:var(--panel); border:1px solid var(--line); border-radius:.65rem;
  max-height:34rem; overflow-y:auto; font-size:.85rem }
.transcript div { padding:.35rem .8rem; border-bottom:1px solid var(--line); cursor:pointer; display:flex; gap:.6rem }
.transcript div:hover { background:var(--hi) }
.transcript div.active { background:var(--hi); border-left:3px solid var(--accent); padding-left:calc(.8rem - 3px) }
.transcript .who { color:var(--accent); font-weight:600; white-space:nowrap; min-width:6.5rem }
.transcript .idx { color:var(--muted); min-width:2.2rem; text-align:right; font-variant-numeric:tabular-nums }
.note { color:var(--muted); font-size:.85rem }
"""

SCRIPT = """
function fmt(s){ if(s==null) return "–"; s=Math.round(s); return Math.floor(s/60)+":"+String(s%60).padStart(2,"0"); }
document.querySelectorAll("section[data-episode]").forEach(section => {
  const data = JSON.parse(section.querySelector("script[type='application/json']").textContent);
  const players = {};
  section.querySelectorAll("audio[data-cond]").forEach(a => players[a.dataset.cond] = a);
  const exclusive = section.querySelector("input.exclusive");
  Object.values(players).forEach(a => a.addEventListener("play", () => {
    if (exclusive.checked) Object.values(players).forEach(o => { if(o!==a) o.pause(); });
  }));
  const rows = Array.from(section.querySelectorAll(".transcript div[data-line]"));
  let cursor = 0;
  function seekAll(i){
    cursor = i;
    rows.forEach(r => r.classList.toggle("active", Number(r.dataset.line)===i));
    for (const [cond, offsets] of Object.entries(data.offsets)) {
      const t = offsets[i], a = players[cond];
      if (a && t != null) a.currentTime = t;
    }
  }
  rows.forEach(r => r.addEventListener("click", () => seekAll(Number(r.dataset.line))));
  section.querySelector("button.pause-all").addEventListener("click",
    () => Object.values(players).forEach(a => a.pause()));
  section.querySelector("button.prev").addEventListener("click", () => { seekAll(Math.max(0,cursor-1)); rows[cursor].scrollIntoView({block:"nearest"}); });
  section.querySelector("button.next").addEventListener("click", () => { seekAll(Math.min(rows.length-1,cursor+1)); rows[cursor].scrollIntoView({block:"nearest"}); });
  // Highlight the line under whichever player is actively playing.
  for (const [cond, a] of Object.entries(players)) {
    a.addEventListener("timeupdate", () => {
      if (a.paused) return;
      const offsets = data.offsets[cond]; if (!offsets) return;
      let best = -1;
      for (let i = 0; i < offsets.length; i++)
        if (offsets[i] != null && offsets[i] <= a.currentTime + 0.05) best = i;
      if (best >= 0 && best !== cursor) { cursor = best; rows.forEach(r => r.classList.toggle("active", Number(r.dataset.line)===best)); }
    });
  }
});
"""


def render(episodes, transcripts, skipped) -> str:
    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>comparativa — side-by-side generations</title>")
    parts.append(f"<style>{STYLE}</style></head><body>")
    parts.append("<h1>comparativa — side-by-side generations</h1>")
    parts.append(
        "<p class='sub'>One player per condition. Click a transcript line to seek "
        "every player to that line on its own timeline; “exclusive” pauses the "
        "others when one plays. Regenerate with "
        "<code>uv run python scripts/comparison_page.py</code>.</p>"
    )

    # Performance summary table.
    parts.append("<table class='perf'><tr><th>cond</th><th>voices / stack</th><th>episode</th>"
                 "<th class='num'>audio</th><th class='num'>wall</th><th class='num'>RTF</th></tr>")
    for episode in sorted(episodes):
        for cond, row in sorted(episodes[episode].items()):
            dur = f"{row.duration/60:.1f} min" if row.duration else "–"
            wall = f"{row.wall/60:.1f} min" if row.wall else "–"
            rtf = f"{row.rtf:.3f}" if row.rtf is not None else "–"
            parts.append(
                f"<tr><td><strong>{cond}</strong></td><td>{html.escape(CONDITION_LABELS[cond])}</td>"
                f"<td>{html.escape(episode)}</td><td class='num'>{dur}</td>"
                f"<td class='num'>{wall}</td><td class='num'>{rtf}</td></tr>"
            )
    parts.append("</table>")
    if skipped:
        names = ", ".join(f"{c} ({CONDITION_LABELS[c]})" for c in skipped)
        parts.append(f"<p class='note'>Skipped conditions: {html.escape(names)} — see bench/&lt;cond&gt;/SKIPPED.md.</p>")

    for episode in sorted(episodes):
        all_rows = episodes[episode]
        # A default-voice render's player is dropped when its cloned sibling
        # was generated for this episode (the perf table above keeps it).
        rows = {
            cond: row
            for cond, row in all_rows.items()
            if SUPERSEDED_BY_CLONE.get(cond) not in all_rows
        }
        transcript = transcripts.get(episode) or []
        offsets = {cond: row.offsets for cond, row in rows.items() if row.offsets}
        parts.append(f"<section data-episode='{html.escape(episode)}'>")
        parts.append(f"<h2>{html.escape(episode)}</h2>")
        parts.append("<div class='controls'>"
                     "<button class='pause-all'>⏸ pause all</button>"
                     "<button class='prev'>◀ line</button><button class='next'>line ▶</button>"
                     "<label><input type='checkbox' class='exclusive' checked> exclusive playback</label>"
                     "</div>")
        parts.append("<div class='episode'><div class='players'>")
        for cond, row in sorted(rows.items()):
            dur = f"{row.duration/60:.1f} min" if row.duration else "–"
            rtf = f"RTF {row.rtf:.2f}" if row.rtf is not None else ""
            sync = "" if row.offsets else " · no line sync"
            parts.append(
                f"<div class='row'><div class='badge'>{cond}</div><div>"
                f"<div class='meta'><strong>{html.escape(CONDITION_LABELS[cond])}</strong>"
                f"<span>{dur} · {rtf}{sync}</span></div>"
                f"<audio controls preload='metadata' data-cond='{cond}' "
                f"src='{html.escape(row.src)}'></audio></div></div>"
            )
        parts.append("</div>")  # players
        parts.append("<div class='transcript'>")
        for i, line in enumerate(transcript):
            who = html.escape(str(line["character"]))
            text = html.escape(str(line["text"]))
            parts.append(f"<div data-line='{i}'><span class='idx'>{i+1}</span>"
                         f"<span class='who'>{who}</span><span>{text}</span></div>")
        parts.append("</div></div>")  # transcript, episode grid
        payload = json.dumps({"offsets": offsets}, separators=(",", ":"))
        parts.append(f"<script type='application/json'>{payload}</script>")
        parts.append("</section>")

    parts.append(f"<script>{SCRIPT}</script></body></html>")
    return "".join(parts)


def main() -> int:
    if not BENCH.is_dir():
        print("no bench/ directory — run `comparativa bench` first")
        return 1
    episodes, _, transcripts, skipped = collect()
    if not episodes:
        print("no completed cells under bench/")
        return 1
    OUT.write_text(render(episodes, transcripts, skipped), encoding="utf-8")
    players = sum(
        1
        for rows in episodes.values()
        for cond in rows
        if SUPERSEDED_BY_CLONE.get(cond) not in rows
    )
    print(f"wrote {OUT} ({len(episodes)} episode(s), {players} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
