#!/bin/bash
# fx-run-build: wait until MemAvailable >= 4.5 GB and no other fx-build runs, then build one scene.
# usage: fx_run_build.sh <scene.bddl> <out_dir>
cd /home/ran/arg_twin
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -ge 4608 ] && ! pgrep -f "^.venv-libero/bin/python -I playgrounds/footage/fx_build" >/dev/null; then break; fi
  echo "$(date +%T) waiting: MemAvailable ${avail} MB" >&2; sleep 5
done
exec .venv-libero/bin/python -I playgrounds/footage/fx_build.py "$1" "$2"
