#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ZARROW_URL="https://github.com/tylitianrui/zarrow/archive/refs/heads/master.tar.gz"
GLOBAL_CACHE_DIR="${GLOBAL_CACHE_DIR:-${ROOT_DIR}/.zig-global-cache}"

HASH="$(zig fetch --global-cache-dir "${GLOBAL_CACHE_DIR}" "${ZARROW_URL}")"
perl -0pi -e 's#(\.url = "https://github.com/tylitianrui/zarrow/archive/refs/heads/master\.tar\.gz",\s*\n\s*\.hash = ")[^"]+(")#$1'"${HASH}"'$2#' build.zig.zon

echo "Updated zarrow hash to: ${HASH}"
