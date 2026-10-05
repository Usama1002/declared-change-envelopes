#!/usr/bin/env bash
# Every table (paper/tables/*.tex) and figure (paper/figures/*.pdf) of the paper from the JSON files in results/.
# Runs on CPU in a few seconds; works with the result files shipped in this repository.
source "$(dirname "$0")/env.sh"

run "$PY" paper/make_tables.py
run "$PY" paper/make_figs.py
