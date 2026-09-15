#!/usr/bin/env bash
# Wheels of the strata packages this one depends on, built from their
# repositories at main into dist/, so that `uv sync --find-links dist`
# resolves them. Inside the strata workspace this is not needed: every
# package there resolves to its sibling checkout. Goes away once the
# packages are on an index.
#
#   .github/sibling-wheels.sh labels common
#
set -euo pipefail
: "${STRATA_GIT:=https://github.com/emaroppo}"
mkdir -p dist .siblings
for name in "$@"; do
    rm -rf ".siblings/$name"
    git clone --quiet --depth 1 "$STRATA_GIT/strata-$name.git" ".siblings/$name"
    uv build --quiet --wheel --out-dir dist ".siblings/$name"
done
