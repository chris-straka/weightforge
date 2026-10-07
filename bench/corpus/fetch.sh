#!/usr/bin/env bash
# Fetch the CC0 inputs and build the benchmark corpus (headless, no GUI).
#   bench/corpus/fetch.sh            -> $FORGE_BENCH/corpus/<character>/*.glb
# FORGE_BENCH defaults to ~/.cache/forge-bench (outside the repo: assets are
# never committed). MPFB2 goes into an isolated Blender profile there, so the
# user's own Blender config is untouched. Sources, licences, checksums:
#   MPFB2 2.0.17 (extensions.blender.org; code GPL-3, data CC0)
#   MakeHuman system assets (static.makehumancommunity.org; CC0)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FB="${FORGE_BENCH:-$HOME/.cache/forge-bench}"
BL="${BLENDER:-$(command -v blender || echo /Applications/Blender.app/Contents/MacOS/Blender)}"
BL="$(readlink -f "$BL")"
mkdir -p "$FB/dl"
get() { # url file sha256
  if [ ! -f "$FB/dl/$2" ] || ! echo "$3  $FB/dl/$2" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fsSL -o "$FB/dl/$2" "$1"
  fi
  echo "$3  $FB/dl/$2" | sha256sum -c --quiet -
}
get "https://extensions.blender.org/download/sha256:4f0a879d64a39bf646fbf5f53601ac678855da329d650617dca5737548239a87/add-on-mpfb-v2.0.17.zip" \
    mpfb.zip 4f0a879d64a39bf646fbf5f53601ac678855da329d650617dca5737548239a87
get "https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip" \
    mh_system_cc0.zip b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107
export BLENDER_USER_RESOURCES="$FB/blender-user"
if [ ! -d "$BLENDER_USER_RESOURCES/extensions/user_default/mpfb" ]; then
  "$BL" --command extension install-file -r user_default -e "$FB/dl/mpfb.zip"
fi
MH_SYSTEM_PACK="$FB/dl/mh_system_cc0.zip" "$BL" -b --factory-startup \
  --python "$HERE/build_mpfb.py" -- "$FB/corpus" "$@" 2>&1 | grep -E '^(BUILT|FAILED)'
