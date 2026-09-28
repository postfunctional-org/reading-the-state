#!/usr/bin/env bash
# Each local model behind serve_local.py on loopback, played over HTTP like every hosted model.
cd "$(dirname "$0")/.." || exit 1
pyfor () { case "$1" in Kai|Lex) echo .venv-enc/bin/python ;; *) echo .venv/bin/python ;; esac; }
mkdir -p logs
for M in Kai Lex Eos Sol Nox Laya Lux; do
  $(pyfor $M) bench/serve_local.py --answerer $M --port 8801 > logs/serve_$M.log 2>&1 &
  SRV=$!
  until grep -q "on 127.0.0.1" logs/serve_$M.log || ! kill -0 $SRV 2>/dev/null; do sleep 3; done
  env ${M^^}_BASE_URL=http://127.0.0.1:8801 ${M^^}_MODEL=$M ${M^^}_PATH=/v1/systemone \
    .venv-tw/bin/python bench/experiments/tw_play/play.py --answerer http:$M 2>&1 | grep -v "^  \["
  kill $SRV; wait $SRV 2>/dev/null
done
