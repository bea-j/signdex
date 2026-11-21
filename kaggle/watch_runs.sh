#!/usr/bin/env bash
# Watch Kaggle training versions; when one finishes, download its outputs to model/runs/kernel-v<N>/ and exit,
# so the caller can report it. Re-run with the remaining versions to keep watching.
#   bash kaggle/watch_runs.sh 600 4 5      # check every 600 s, versions 4 and 5
set -u
interval=$1; shift
cd "$(dirname "$0")/.."
while true; do
  for v in "$@"; do
    dir="model/runs/kernel-v$v"
    [ -f "$dir/.done" ] && continue
    st=$(kaggle kernels status "beatricej/signseek-train/$v" 2>&1 | tail -1 | cut -c1-160)
    echo "$(date +%H:%M:%S) v$v: $st"
    case "$st" in
      *COMPLETE*)
        mkdir -p "$dir"
        kaggle kernels output "beatricej/signseek-train/$v" -p "$dir" -o 2>&1 | tail -1 | cut -c1-160
        ls "$dir"
        touch "$dir/.done"
        echo "FINISHED v$v"
        exit 0 ;;
      *ERROR*|*CANCEL*)
        mkdir -p "$dir"
        kaggle kernels output "beatricej/signseek-train/$v" -p "$dir" -o 2>&1 | tail -1 | cut -c1-160
        touch "$dir/.done"
        echo "FAILED v$v: $st"
        exit 0 ;;
    esac
  done
  pending=0
  for v in "$@"; do [ -f "model/runs/kernel-v$v/.done" ] || pending=1; done
  [ $pending -eq 0 ] && { echo "all done"; exit 0; }
  sleep "$interval"
done
