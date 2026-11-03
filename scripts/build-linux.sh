#!/bin/sh
# Usage: sh scripts/build-linux.sh [cpu|cuda|both] [CMake options...]
set -eu
if [ "$(uname -s)" != Linux ]; then
    printf 'Use build-macos.sh on macOS or build-windows.ps1 on Windows.\n' >&2
    exit 2
fi
backend=${1:-cuda}
if [ "$#" -gt 0 ]; then shift; fi
exec sh "$(dirname -- "$0")/build-unix.sh" "$backend" "$@"
