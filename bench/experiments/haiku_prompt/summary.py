"""Coached vs neutral Haiku: what the task-specific prompt was worth.

coached  bench/answers-haiku-coached/  the originally published run; its prompts added
         task knowledge the decision models never receive (board legend, "look/examine
         waste a turn", the roguelike's three position types, "every probe square is provable")
neutral  bench/answers/                format rules + each position's own instructions only
b28      tw_history/haiku_b28__*.json  neutral wording, but 28 positions per agent and no
         structured output - kept because it shows how much the harness alone moves Haiku
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from spec import ENVS, POSITIONS, load_positions  # noqa: E402


def rate(picks, pos):
    return sum(picks.get(p["pid"]) in p["truth"]["optimal_ids"] for p in pos) / len(pos)


def cert(said, truth):
    s = [v for q, v in said.items() if not truth[q]]; m = [v for q, v in said.items() if truth[q]]
    au = sum((a > b) + 0.5 * (a == b) for a in s for b in m) / (len(s) * len(m))
    return {"gap": sum(s) / len(s) - sum(m) / len(m), "auroc": au}


out = {"tasks": {}}
for env in ENVS:
    pos = load_positions(env)
    c = json.loads((BENCH / "answers-haiku-coached" / f"{env}__Haiku-4.5.json").read_text())["picks"]
    n = json.loads((BENCH / "answers" / f"{env}__Haiku-4.5.json").read_text())["picks"]
    out["tasks"][env] = {"coached": rate(c, pos), "neutral": rate(n, pos),
                         "same_pick": sum(c.get(p["pid"]) == n.get(p["pid"]) for p in pos) / len(pos)}
truth = json.loads((POSITIONS / "certainty.truth.json").read_text())
out["certainty"] = {k: cert(json.loads((BENCH / d / "certainty__Haiku-4.5.json").read_text())["said"], truth)
                    for k, d in (("coached", "answers-haiku-coached"), ("neutral", "answers"))}
tw = {p["pid"]: p for p in load_positions("textworld")}
b28 = json.loads((HERE.parent / "tw_history" / "haiku_b28__A.json").read_text())["picks"]
out["textworld_b28"] = {"rate": rate(b28, list(tw.values())),
                        "invalid": sum(1 for k, v in b28.items() if v not in {c["id"] for c in tw[k]["candidates"]})}
(HERE / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
