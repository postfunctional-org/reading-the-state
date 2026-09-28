#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
R=bench/experiments/ms_structured/run.py
python3 $R --answerer heuristic
for M in Kai Lex; do .venv-enc/bin/python $R --answerer $M 2>&1 | grep "^\[" ; done
for M in Eos Sol Nox Laya Lux; do .venv/bin/python $R --answerer $M 2>&1 | grep "^\[" ; done
