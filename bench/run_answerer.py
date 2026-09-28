"""Run one answerer over one environment's frozen positions and save its picks.

    python bench/run_answerer.py --answerer Nox --env minesweeper
    python bench/run_answerer.py --answerer heuristic --env all
    JEV_BASE_URL=... JEV_API_KEY=... python bench/run_answerer.py --answerer http:JEV --env all

Positions are frozen on disk, so every answerer sees byte-identical inputs and the
comparison is paired position by position. Answers are written separately from
positions; the grader joins them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import ANSWERS, ENVS, load_positions  # noqa: E402


def run(answerer, env: str, limit: int | None = None) -> dict:
    positions = load_positions(env)
    if limit:
        positions = positions[:limit]
    picks, telem, errors = {}, {}, {}
    t0 = time.time()
    for i, p in enumerate(positions):
        try:
            choice, info = answerer.pick(p)
            picks[p["pid"]] = choice
            if info:
                telem[p["pid"]] = info
        except Exception as e:                      # never let one position kill a run
            errors[p["pid"]] = f"{type(e).__name__}: {str(e)[:160]}"
        if (i + 1) % 25 == 0:
            print(f"  [{answerer.name}/{env}] {i+1}/{len(positions)}"
                  f"  {time.time()-t0:.0f}s", flush=True)
    return {
        "answerer": answerer.name,
        "kind": getattr(answerer, "kind", "model"),
        "env": env,
        "n_positions": len(positions),
        "picks": picks,
        "telemetry": telem,
        "errors": errors,
        "wall_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", required=True,
                    help="Kai|Lex|Eos|Sol|Nox|Laya|oracle|uniform-random|heuristic|http:NAME")
    ap.add_argument("--env", default="all", choices=("all",) + ENVS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import answerers

    a = answerers.build(args.answerer)
    envs = ENVS if args.env == "all" else (args.env,)
    ANSWERS.mkdir(parents=True, exist_ok=True)
    for env in envs:
        try:
            load_positions(env)
        except FileNotFoundError:
            print(f"  [{a.name}] no positions for {env}, skipping", flush=True)
            continue
        res = run(a, env, args.limit)
        safe = a.name.replace("/", "_")
        out = ANSWERS / f"{env}__{safe}.json"
        out.write_text(json.dumps(res, indent=1))
        n_err = len(res["errors"])
        print(f"[{a.name}/{env}] {len(res['picks'])} answered"
              f"{f', {n_err} errors' if n_err else ''}"
              f"  {res['wall_seconds']}s -> {out.name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
