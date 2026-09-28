"""Jev is not deterministic. Re-run the full benchmark and the certainty probe
under the same settings, into this folder, so the headline run (bench/answers/)
can be reported with its run-to-run spread. Never writes to bench/answers/.

    JEV_BASE_URL=... JEV_API_KEY=... JEV_MODEL=jev-1.13.0 python bench/experiments/jev_repeats/run.py 2 3
"""
import json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import answerers                                   # noqa: E402
from run_answerer import run                       # noqa: E402
from run_certainty import ask, build_probes        # noqa: E402
from spec import ENVS                              # noqa: E402

a = answerers.build("http:Jev")
for r in sys.argv[1:]:
    for env in ENVS:
        res = run(a, env)
        (HERE / f"{env}__Jev.r{r}.json").write_text(json.dumps(res, indent=1))
        print(f"[r{r}/{env}] {len(res['picks'])} answered, {len(res['errors'])} errors", flush=True)
    said, errors = {}, {}
    for p in build_probes():
        try:
            said[p["qid"]] = ask(a, p)
        except Exception as e:
            errors[p["qid"]] = f"{type(e).__name__}: {str(e)[:140]}"
    (HERE / f"certainty__Jev.r{r}.json").write_text(json.dumps({"said": said, "errors": errors}, indent=1))
    print(f"[r{r}/certainty] {len(said)} answered, {len(errors)} errors", flush=True)
