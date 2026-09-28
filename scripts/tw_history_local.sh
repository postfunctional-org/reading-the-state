#!/usr/bin/env bash
# Arms A and B for every local answerer, same session, same code path.
cd "$(dirname "$0")/.." || exit 1
R=bench/experiments/tw_history/run.py
python3 $R --answerer heuristic --arms AB 2>&1 | tail -2
for M in Kai Lex; do .venv-enc/bin/python $R --answerer $M --arms AB 2>&1 | grep "^\[" ; done
for M in Eos Sol Nox Laya Lux; do .venv/bin/python $R --answerer $M --arms AB 2>&1 | grep "^\[" ; done
