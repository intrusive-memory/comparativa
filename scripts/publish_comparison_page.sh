#!/usr/bin/env bash
# Publish bench/comparison.html (plus exactly the media it references) to the
# repo's GitHub Pages site, served from the `gh-pages` branch.
#
#   scripts/publish_comparison_page.sh
#
# The site is assembled in a temp directory and force-pushed as a single
# orphan commit, so the audio artifacts never enter main's history (bench/ is
# gitignored on purpose). Enable Pages once with:
#
#   gh api --method POST repos/{owner}/{repo}/pages \
#     -f 'source[branch]=gh-pages' -f 'source[path]=/'
#
# Re-running this script replaces the published site with the current page.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO_ROOT/bench"
PAGE="$BENCH/comparison.html"
REMOTE="$(git -C "$REPO_ROOT" remote get-url origin)"

# 1. Regenerate the page so the site always matches the current bench/.
uv run --project "$REPO_ROOT" python "$REPO_ROOT/scripts/comparison_page.py"

# 2. Assemble the site: index.html + only the media the page references.
SITE="$(mktemp -d /tmp/comparativa-pages.XXXXXX)"
trap 'rm -rf "$SITE"' EXIT
cp "$PAGE" "$SITE/index.html"
touch "$SITE/.nojekyll"

count=0
while IFS= read -r rel; do
  src="$BENCH/$rel"
  if [[ ! -f "$src" ]]; then
    echo "error: page references missing media: $rel" >&2
    exit 1
  fi
  mkdir -p "$SITE/$(dirname "$rel")"
  cp "$src" "$SITE/$rel"
  count=$((count + 1))
done < <(grep -o "src='[^']*'" "$PAGE" | sed "s/^src='//;s/'$//" | sort -u)

echo "site: index.html + $count media file(s), $(du -sh "$SITE" | cut -f1)"

# 3. One orphan commit, force-pushed to gh-pages.
git -C "$SITE" init -q -b gh-pages
git -C "$SITE" add -A
git -C "$SITE" -c user.name="comparativa" -c user.email="noreply@intrusive-memory.productions" \
  commit -q -m "Publish comparison page ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
git -C "$SITE" push -f "$REMOTE" gh-pages:gh-pages
echo "pushed gh-pages to $REMOTE"
