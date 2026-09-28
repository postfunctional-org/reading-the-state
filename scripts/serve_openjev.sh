#!/usr/bin/env bash
# OpenJev exactly as it ran for every published OpenJev number (flags captured from the
# live processes on the RTX 3090 box, 2026-09-27). Loopback only: the helper has no auth.
# MODELS holds openjev-FP8/ and the vendor's openjev-helper/ checkout.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; MODELS="${MODELS:-$HOME/models}"
cd "$ROOT" || exit 1; mkdir -p logs
VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-jev/bin/vllm serve $MODELS/openjev-FP8 \
  --host 127.0.0.1 --port 8111 --served-model-name qwen --enable-prefix-caching \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 8 --max-logprobs 64 \
  --gpu-memory-utilization 0.92 --cpu-offload-gb 12 --offload-backend uva \
  --gdn-prefill-backend triton --limit-mm-per-prompt '{"image":1}' --trust-remote-code \
  > logs/openjev_vllm.log 2>&1 &
until curl -sf http://127.0.0.1:8111/v1/models >/dev/null; do sleep 5; done
cd $MODELS/openjev-helper || exit 1
VLLM=http://localhost:8111/v1 TOKENIZER=$MODELS/openjev-FP8 \
READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 READOUT_TARGETED=1 \
READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
nohup $ROOT/.venv-jev/bin/python helper/shim.py --host 127.0.0.1 --port 3111 \
  > $ROOT/logs/openjev_shim.log 2>&1 &
# verify: curl -s http://127.0.0.1:3111/v1/version  -> shim_sha256 81a22f1b...
