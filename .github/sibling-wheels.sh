#!/usr/bin/env bash
# Wheels of the strata packages this one depends on, built from their
# repositories at main into dist/, so that `uv sync --find-links dist`
# resolves them. Inside the strata workspace this is not needed: every
# package there resolves to its sibling checkout.
#
#   .github/sibling-wheels.sh contracts common
#
# From GitHub unless STRATA_GIT names another base URL the repositories sit
# under, a mirror say:  STRATA_GIT=http://git.example.lan:3000/you
#
set -euo pipefail
: "${STRATA_GIT:=https://github.com/emaroppo}"
mkdir -p dist .siblings
for name in "$@"; do
    rm -rf ".siblings/$name"
    git clone --quiet --depth 1 "$STRATA_GIT/strata-$name.git" ".siblings/$name"
    uv build --quiet --wheel --out-dir dist ".siblings/$name"
done
