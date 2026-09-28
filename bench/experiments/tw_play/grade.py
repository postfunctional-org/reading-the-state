"""Grade the playthroughs and write summary.json for the report.

Per answerer, over the same 59 games:
  won        games finished with the objective complete, Wilson 95% interval
  moves/opt  on won games, moves used / optimal moves (1.00 = perfect)
  progress   on lost games, share of the starting distance to the goal it closed
  loops      games that ran out of moves while repeating one short cycle of commands
Paired: exact McNemar on won/lost per game against Jev.
Also: step-level accuracy with history (tw_history arm B, all 224 positions), for
comparison with how that per-step rate compounds over a whole game.
"""
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from spec import fisher_exact_two_sided, wilson  # noqa: E402

ORDER = ["Jev", "OpenJev", "CLM", "Laya", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex", "Haiku-4.5",
         "heuristic", "uniform-random", "oracle"]


def mcnemar(a, b):
    n01 = sum(1 for k in a if not a[k] and b[k]); n10 = sum(1 for k in a if a[k] and not b[k]); n = n01 + n10
    return n01, n10, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n))


def looped(g):
    """ran out of moves and the last 6 moves use at most 2 distinct commands"""
    return (not g["won"]) and g["moves"] >= g["budget"] and len(set(g["actions"][-6:])) <= 2


def main():
    hist = json.loads((HERE.parent / "tw_history" / "summary.json").read_text())
    stepB = {r["name"]: r["all"]["B"]["rate"] for r in hist["rows"]}
    res = {}
    for n in ORDER:
        f = HERE / "results" / f"{n}.json"
        if f.exists():
            res[n] = {g["seed"]: g for g in json.loads(f.read_text())["games"]}
    jev = {k: g["won"] for k, g in res["Jev"].items()}
    # the bar: the best player that reads nothing (uniform random; the heuristic has no memory and loops)
    bw = max(sum(g["won"] for g in res[b].values()) for b in ("uniform-random", "heuristic") if b in res)
    bn = len(res["Jev"])
    rows = []
    for n, G in res.items():
        gs = list(G.values()); w = sum(g["won"] for g in gs); lo, hi = wilson(w, len(gs))
        wins = [g for g in gs if g["won"]]; losses = [g for g in gs if not g["won"]]
        prog = [(g["distance"][0] - (g["distance"][-1] if g["distance"][-1] is not None else g["distance"][0]))
                / g["distance"][0] for g in losses]
        r = {"name": n, "n": len(gs), "won": w, "rate": w / len(gs), "ci": [lo, hi],
             "moves_over_opt": (sum(g["moves"] / g["oracle_len"] for g in wins) / len(wins)) if wins else None,
             "loss_progress": (sum(prog) / len(prog)) if prog else None,
             "loops": sum(looped(g) for g in gs), "step_acc_with_history": stepB.get(n),
             "vs_baseline_p": fisher_exact_two_sided(w, len(gs) - w, bw, bn - bw)}
        if n != "Jev":
            other = {k: g["won"] for k, g in G.items()}
            f01, f10, p = mcnemar(jev, other)
            r["vs_jev"] = {"only_this": f01, "only_jev": f10, "p": p}
        if "invalid_attempts" in gs[0]:
            r["invalid_attempts"] = sum(g.get("invalid_attempts", 0) for g in gs)
        rows.append(r)
    out = {"n_games": len(res["Jev"]), "baseline_won": bw, "budget": "3x optimal moves, at least 15",
           "oracle_len": sorted({g["oracle_len"] for g in res["Jev"].values()}), "rows": rows}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{'':15s} {'won':>7s} {'95% CI':>12s} {'moves/opt':>9s} {'loss prog':>9s} {'loops':>5s} {'step acc':>8s}  vs Jev")
    for r in rows:
        v = r.get("vs_jev")
        print(f"{r['name']:15s} {r['won']:3d}/{r['n']:<3d} [{r['ci'][0]:4.0%},{r['ci'][1]:4.0%}] "
              f"{(r['moves_over_opt'] or 0):9.2f} {(r['loss_progress'] if r['loss_progress'] is not None else 0):9.0%} "
              f"{r['loops']:5d} {(r['step_acc_with_history'] or 0):8.1%}"
              + (f"  +{v['only_this']}/-{v['only_jev']} p={v['p']:.3g}" if v else ""))


if __name__ == "__main__":
    main()
