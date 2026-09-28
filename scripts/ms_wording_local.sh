#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
for M in Kai Lex; do for D in diag diag_mine; do .venv-enc/bin/python bench/experiments/ms_wording/$D.py $M 2>&1 | tail -1; done; done
for M in Eos Sol Nox Laya Lux; do for D in diag diag_mine; do .venv/bin/python bench/experiments/ms_wording/$D.py $M 2>&1 | tail -1; done; done
