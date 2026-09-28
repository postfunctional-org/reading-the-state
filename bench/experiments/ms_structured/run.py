"""Run one answerer over the structured Minesweeper positions (same code path as
bench/run_answerer.py: answerers.build + pick, errors recorded, never aborting).

    .venv/bin/python bench/experiments/ms_structured/run.py --answerer Nox
"""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import answerers  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--answerer", required=True); args = ap.parse_args()
a = answerers.build(args.answerer)
skip = set(json.loads((HERE / "excluded.json").read_text())["excluded"])
pos = [p for p in json.loads((HERE / "minesweeper_structured.json").read_text()) if p["pid"] not in skip]
picks, telem, errors, t0 = {}, {}, {}, time.time()
for p in pos:
    try:
        c, info = a.pick(p); picks[p["pid"]] = c
        if info: telem[p["pid"]] = info
    except Exception as e:
        errors[p["pid"]] = f"{type(e).__name__}: {str(e)[:160]}"
(HERE / f"answers__{a.name}.json").write_text(json.dumps(
    {"answerer": a.name, "picks": picks, "telemetry": telem, "errors": errors, "wall_seconds": round(time.time() - t0, 1)}, indent=1))
print(f"[{a.name}] {len(picks)} answered, {len(errors)} errors", flush=True)
