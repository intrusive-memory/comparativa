#!/usr/bin/env bash
#
# bench_condition_a.sh — OPERATION BATTLING BARDS, Sortie 8, condition A.
#
# Regenerates the Swift baseline (Produciesta / SwiftVoxAlta) audio FRESH for
# the corpus episodes, timing wall-clock and peak RSS with /usr/bin/time -l.
#
# NO BUILDS. This script invokes only the pre-existing signed binary at
# ~/.local/bin/produciesta. It never compiles anything.
#
# The granville project tree is READ-ONLY input:
#   * every output path (--out, its sidecar .vtt) is inside bench/A/
#   * --cache-dir is redirected to a per-run scratch dir, so the tool's
#     ephemeral store never lands in Application Support or in the project
#   * the 2026-08-09 audio/*.m4a reference artifacts are never touched
#
# Usage:
#   scripts/bench_condition_a.sh [episode-stem ...]
#
# Environment overrides:
#   PRODUCIESTA   path to the signed binary   (default: ~/.local/bin/produciesta)
#   PROJECT_DIR   granville project root      (default: ~/Projects/podcasts/granville)
#   BENCH_DIR     condition-A output root     (default: <repo>/bench/A)
#   SCRATCH_DIR   parent for the run caches   (default: $TMPDIR)
#   CACHE_MODE    scratch | default           (default: scratch)
#   FORCE         1 = re-run a cell that already has audio
#
# CACHE_MODE=scratch passes --cache-dir <mktemp -d>, which is the hermetic
# choice and what the bumper was measured with. CACHE_MODE=default omits the
# flag entirely, letting Produciesta use ~/Library/Application Support/
# Produciesta — still outside the granville tree, and the configuration the
# 2026-08-14 production renders used. Long episodes have been seen to fail at
# the compose stage with `error[adapter-error]` (the pipeline's SwiftData store
# failing to read generated audio back), so the fallback is worth having.
# A failed run's cache dir is NOT deleted, so it can be inspected.
#
# Outputs, per episode, under $BENCH_DIR/<episode>/:
#   <episode>.m4a         the composed episode (Produciesta's native export)
#   <episode>.vtt         Produciesta's sidecar transcript (one cue per Element)
#   generate.log          the child's stdout (NDJSON progress events)
#   time.txt              the child's stderr + the /usr/bin/time -l block
#   run.json              command, exit code, timestamps, versions
#
# metrics.json is written afterwards by scripts/bench_condition_a_metrics.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCIESTA="${PRODUCIESTA:-$HOME/.local/bin/produciesta}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/Projects/podcasts/granville}"
BENCH_DIR="${BENCH_DIR:-$REPO_ROOT/bench/A}"
SCRATCH_DIR="${SCRATCH_DIR:-${TMPDIR:-/tmp}}"
CACHE_MODE="${CACHE_MODE:-scratch}"
FORCE="${FORCE:-0}"

DEFAULT_EPISODES=(
  episode_1_01_cold_open
  episode_1_01a_bumper_donnie_and_arnie_1
)

if [[ $# -gt 0 ]]; then
  EPISODES=("$@")
else
  EPISODES=("${DEFAULT_EPISODES[@]}")
fi

[[ -x "$PRODUCIESTA" ]] || { echo "error: produciesta not executable at $PRODUCIESTA" >&2; exit 2; }
[[ -d "$PROJECT_DIR" ]] || { echo "error: project dir not found: $PROJECT_DIR" >&2; exit 2; }

PRODUCIESTA_VERSION="$("$PRODUCIESTA" --version)"
IDENTITY="$("$PRODUCIESTA" version 2>&1 || true)"
VOXALTA_VERSION="$(printf '%s\n' "$IDENTITY" | awk '/^swift-voxalta:/ {print $2}')"
SPEC_HASH="$(printf '%s\n' "$IDENTITY" | awk '/^spec-hash:/ {print $2}')"
MACOS_VERSION="$(sw_vers -productVersion)"

echo "condition A — Swift baseline (Produciesta $PRODUCIESTA_VERSION / SwiftVoxAlta ${VOXALTA_VERSION:-?})"
echo "  binary:  $PRODUCIESTA -> $(readlink "$PRODUCIESTA" 2>/dev/null || echo "$PRODUCIESTA")"
echo "  project: $PROJECT_DIR (read-only)"
echo "  output:  $BENCH_DIR"
echo

rc_total=0

for ep in "${EPISODES[@]}"; do
  screenplay="$PROJECT_DIR/episodes/$ep.fountain"
  [[ -f "$screenplay" ]] || { echo "error: no screenplay for '$ep' at $screenplay" >&2; rc_total=1; continue; }

  outdir="$BENCH_DIR/$ep"
  out_m4a="$outdir/$ep.m4a"

  if [[ -f "$out_m4a" && "$FORCE" != "1" ]]; then
    echo "[$ep] skip — $out_m4a exists (FORCE=1 to re-run)"
    continue
  fi

  mkdir -p "$outdir"
  rm -f "$out_m4a" "$outdir/$ep.wav" "$outdir/$ep.vtt" "$out_m4a.vtt" \
        "$outdir/generate.log" "$outdir/time.txt" "$outdir/run.json" "$outdir/metrics.json"

  cache_args=()
  cache_dir=""
  if [[ "$CACHE_MODE" == "scratch" ]]; then
    cache_dir="$(mktemp -d "$SCRATCH_DIR/produciesta-bench-A-XXXXXX")"
    cache_args=(--cache-dir "$cache_dir")
  fi
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "[$ep] generating -> $out_m4a (cache: ${cache_dir:-produciesta default})"
  set +e
  /usr/bin/time -l "$PRODUCIESTA" export "$screenplay" \
    --out "$out_m4a" \
    --json \
    --project-md "$PROJECT_DIR/PROJECT.md" \
    --voices-dir "$PROJECT_DIR" \
    ${cache_args[@]+"${cache_args[@]}"} \
    >"$outdir/generate.log" 2>"$outdir/time.txt"
  rc=$?
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -n "$cache_dir" && $rc -eq 0 ]]; then
    rm -rf "$cache_dir"
  elif [[ -n "$cache_dir" ]]; then
    echo "[$ep] cache dir kept for inspection: $cache_dir" >&2
  fi

  cat >"$outdir/run.json" <<JSON
{
  "condition": "A",
  "stack": "swift",
  "episode": "$ep",
  "episode_path": "$screenplay",
  "command": "/usr/bin/time -l $PRODUCIESTA export $screenplay --out $out_m4a --json --project-md $PROJECT_DIR/PROJECT.md --voices-dir $PROJECT_DIR ${cache_dir:+--cache-dir <scratch>}",
  "cache_mode": "$CACHE_MODE",
  "returncode": $rc,
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "produciesta_version": "$PRODUCIESTA_VERSION",
  "swift_voxalta_version": "$VOXALTA_VERSION",
  "spec_hash": "$SPEC_HASH",
  "macos": "$MACOS_VERSION"
}
JSON

  if [[ $rc -ne 0 ]]; then
    echo "[$ep] FAILED (exit $rc) — see $outdir/time.txt" >&2
    rc_total=1
    continue
  fi

  echo "[$ep] ok — $(du -h "$out_m4a" | cut -f1), $(awk '/ real +/ {w=$1} END {print w" s wall"}' "$outdir/time.txt")"
done

echo
echo "done. Now write metrics with:"
echo "  uv run python scripts/bench_condition_a_metrics.py"
exit $rc_total
