#!/usr/bin/env bash
# The only command a Haiku agent may run during whole-game play: show the turn, or play a move.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/.venv-tw/bin/python" "$ROOT/bench/experiments/tw_play/turn.py" "$@"
