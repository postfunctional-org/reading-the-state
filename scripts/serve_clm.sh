#!/usr/bin/env bash
# clm-serve on loopback only (its --host defaults to 0.0.0.0 with auth off).
cd "$(dirname "$0")/.." || exit 1
exec .venv-clm/bin/clm-serve --host 127.0.0.1 --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings \
  --emb-model qwen3-8b --no-ui "$@"
