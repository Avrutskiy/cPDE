#!/bin/sh
# Shared macOS/Linux build driver. No downloads or Python-side compilation.
set -eu
backend=${1:?Usage: build-unix.sh cpu|cuda|both [CMake options...]}
shift
case "$backend" in cpu|cuda|both) ;; *) printf 'Unknown backend: %s\n' "$backend" >&2; exit 2 ;; esac
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir=${CPDE_BUILD_DIR:-"$root/build-$backend"}
jobs=${CPDE_BUILD_JOBS:-2}
cmake -S "$root" -B "$build_dir" -DCPDE_BACKEND="$backend" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$build_dir" --config Release --parallel "$jobs"
printf 'Built %s backend in %s\n' "$backend" "$build_dir"
