#!/usr/bin/env bash
# Qwen3-8B pooling encoder for clm-serve, with the flags of the vendor's own
# serve_qwen3_8b.sh (enforce-eager, prefix cache, max-num-seqs 32; utilisation raised from 0.35, which leaves no KV room on a 24 GB card), bound to loopback.
cd "$(dirname "$0")/.." || exit 1
exec .venv-clm/bin/vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \
  --enforce-eager --enable-prefix-caching --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 --host 127.0.0.1 --port 8090
