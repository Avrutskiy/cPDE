#!/bin/sh
# Native architecture (Apple Silicon or Intel); Accelerate is detected by CMake.
set -eu
if [ "$(uname -s)" != Darwin ]; then
    printf 'Use build-linux.sh on Linux or build-windows.ps1 on Windows.\n' >&2
    exit 2
fi
exec sh "$(dirname -- "$0")/build-unix.sh" cpu "$@"
