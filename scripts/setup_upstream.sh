#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/external/wave-mprage"
URL="https://github.com/HarmonizedMRI/wave-mprage.git"

if [[ -e "$TARGET" ]]; then
    echo "Upstream path already exists: $TARGET"
    exit 0
fi

if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$ROOT" submodule add "$URL" external/wave-mprage
    git -C "$ROOT" submodule update --init --recursive
else
    git clone --recurse-submodules "$URL" "$TARGET"
fi

echo "Upstream wave-mprage is available at: $TARGET"
