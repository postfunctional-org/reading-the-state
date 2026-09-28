"""Does a model's Minesweeper pick follow the direction of the question?
Three wordings of the same positions: the frozen "least likely to be a mine"
(published answers), "most likely to be safe" (same meaning, no negation), and
"most likely to be a mine" (the opposite goal). A model that reads the goal should
move toward mines on the third; one that does not will barely change."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from spec import load_positions
P = load_positions("minesweeper")
rows = []
for m in ["Jev", "OpenJev", "CLM", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex", "Laya"]:
    fs, fm = HERE / f"diag__{m}.json", HERE / f"diag_mine__{m}.json"
    if not (fs.exists() and fm.exists()):
        continue
    A = json.loads((HERE.parents[1] / "answers" / f"minesweeper__{m}.json").read_text())["picks"]
    S = json.loads(fs.read_text())["picks"]; M = json.loads(fm.read_text())["picks"]
    def mine_rate(pk):  # mean true mine probability of the picked square
        return sum(p["truth"]["extra"]["p_mine"][pk[p["pid"]]] for p in P) / len(P)
    rows.append({"name": m, "p_mine_least": mine_rate(A), "p_mine_safe": mine_rate(S), "p_mine_mine": mine_rate(M),
                 "same_pick_when_goal_reversed": sum(A[p["pid"]] == M[p["pid"]] for p in P) / len(P)})
rnd = sum(sum(p["truth"]["extra"]["p_mine"][c["id"]] for c in p["candidates"]) / len(p["candidates"]) for p in P) / len(P)
out = {"uniform_p_mine": rnd, "rows": rows}
(HERE / "summary.json").write_text(json.dumps(out, indent=1))
print(f"mean true P(mine) of the picked square (uniform random {rnd:.2f})")
print(f"{'':6s} {'least mine':>10s} {'most safe':>10s} {'MOST MINE':>10s}   same pick when goal reversed")
for r in rows:
    print(f"{r['name']:6s} {r['p_mine_least']:10.2f} {r['p_mine_safe']:10.2f} {r['p_mine_mine']:10.2f}   {r['same_pick_when_goal_reversed']:.0%}")
