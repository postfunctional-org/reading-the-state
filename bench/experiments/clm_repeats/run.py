"""Second full CLM pass (tasks + certainty) into this folder, never bench/answers/,
so run-to-run agreement can be measured the same way as Jev's.

    CLM_BASE_URL=http://127.0.0.1:8700 CLM_MODEL=clm-latest python bench/experiments/clm_repeats/run.py 2
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import answerers                                  # noqa: E402
from run_answerer import run                      # noqa: E402
from run_certainty import ask, build_probes       # noqa: E402
from spec import ENVS                             # noqa: E402

a = answerers.build("http:CLM")
for r in sys.argv[1:]:
    for env in ENVS:
        res = run(a, env)
        (HERE / f"{env}__CLM.r{r}.json").write_text(json.dumps(res, indent=1))
        print(f"[r{r}/{env}] {len(res['picks'])} answered, {len(res['errors'])} errors", flush=True)
    said, errors = {}, {}
    for p in build_probes():
        try:
            said[p["qid"]] = ask(a, p)
        except Exception as e:
            errors[p["qid"]] = f"{type(e).__name__}: {str(e)[:140]}"
    (HERE / f"certainty__CLM.r{r}.json").write_text(json.dumps({"said": said, "errors": errors}, indent=1))
    print(f"[r{r}/certainty] {len(said)} answered, {len(errors)} errors", flush=True)
