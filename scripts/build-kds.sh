#!/usr/bin/env bash
# Rebuilds ./kds_server as a Release build of the pinned ckdbs commit, in a private clone under var/kds-src/.
#   scripts/build-kds.sh [source-repo] [commit]
# The pin is the commit var/kds/xrock.db was created with (superblock v17); a build with another superblock version
# refuses to mount it. Stop the server first (scripts/kds.sh stop); the previous binary is kept as kds_server.prev.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${1:-/home/cdkbs/ckdbs}"
COMMIT="${2:-25c58497}"
SRC="$ROOT/var/kds-src"

if [[ -d "$SRC/.git" ]]; then
  git -C "$SRC" fetch -q origin
else
  git clone -q --no-checkout "$REPO" "$SRC"
fi
git -C "$SRC" checkout -q --detach "$COMMIT"
cmake -S "$SRC" -B "$SRC/build-release" -DCMAKE_BUILD_TYPE=Release -DKDS_BUILD_TESTS=OFF >/dev/null
cmake --build "$SRC/build-release" -j"$(nproc)" --target kds_server
[[ -f "$ROOT/kds_server" ]] && mv "$ROOT/kds_server" "$ROOT/kds_server.prev"
cp "$SRC/build-release/kds_server" "$ROOT/kds_server"
echo "kds_server: Release build of $(git -C "$SRC" log -1 --format='%h %ci')"
