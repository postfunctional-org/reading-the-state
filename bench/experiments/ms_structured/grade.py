"""Grade structured-state Minesweeper against each answerer's published ASCII-grid answers.

The frozen ASCII answers are arm A (for Jev, its published first run); the structured
state is arm S. Same positions, candidates, instructions and truth, so every
comparison is paired per position (exact McNemar). Also: the hard half (positions
the state-blind 'always corner' gets wrong) and the share of provable-square
landings that are mines (uniform random: 76%).
"""
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from spec import load_positions, wilson  # noqa: E402

ORDER = ["Jev", "OpenJev", "CLM", "Laya", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex", "Haiku-4.5", "heuristic"]


def mcnemar(a, b):
    n01 = sum(1 for k in a if not a[k] and b[k]); n10 = sum(1 for k in a if a[k] and not b[k]); n = n01 + n10
    return n01, n10, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n))


def main():
    skip = set(json.loads((HERE / "excluded.json").read_text())["excluded"])
    P = {p["pid"]: p for p in load_positions("minesweeper") if p["pid"] not in skip}
    brk = json.loads((BENCH / "breakdown.json").read_text())["minesweeper"]["__meta__"]
    triv = brk["best_trivial"]["kind"]
    hard = {k for k, p in P.items() if not any(c["kind"] == triv and c["id"] in p["truth"]["optimal_ids"] for c in p["candidates"])}
    rows = []
    for n in ORDER:
        fs = HERE / f"answers__{n}.json"
        if not fs.exists():
            print(f"  ({n}: no structured run yet)"); continue
        S = json.loads(fs.read_text()); A = json.loads((BENCH / "answers" / f"minesweeper__{n}.json").read_text())["picks"]
        ok = lambda pk, k: pk.get(k) in P[k]["truth"]["optimal_ids"]
        r = {"name": n, "errors": len(S["errors"]),
             "invalid": sum(1 for k, v in S["picks"].items() if v not in {c["id"] for c in P[k]["candidates"]})}
        for lab, sub in (("all", set(P)), ("hard", hard)):
            a = {k: ok(A, k) for k in sub}; s = {k: ok(S["picks"], k) for k in sub}
            f, b, p = mcnemar(a, s)
            r[lab] = {"n": len(sub), "A": sum(a.values()) / len(sub), "S": sum(s.values()) / len(sub),
                      "S_ci": list(wilson(sum(s.values()), len(sub))), "fixed": f, "broken": b, "p": p}
        pm = [P[k]["truth"]["extra"]["p_mine"].get(v) for k, v in S["picks"].items()]
        prov = [x for x in pm if x in (0.0, 1.0)]
        r["provable"] = len(prov); r["mine_share"] = (sum(prov) / len(prov)) if prov else None
        rows.append(r)
    out = {"n": len(P), "excluded": sorted(skip), "n_hard": len(hard), "trivial": brk["best_trivial"]["policy"],
           "trivial_rate": brk["best_trivial"]["rate"], "chance_hard": brk["chance_hard"], "rows": rows}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{'':11s} {'grid':>6s} {'struct':>6s}  fixed/broken  p        {'hard grid':>9s} {'hard struct':>11s}  p      mines/provable")
    for r in rows:
        a, h = r["all"], r["hard"]
        print(f"{r['name']:11s} {a['A']:6.1%} {a['S']:6.1%}  +{a['fixed']:3d}/-{a['broken']:<3d}  {a['p']:<8.2g} {h['A']:9.1%} {h['S']:11.1%}  {h['p']:<6.2g} "
              f"{r['mine_share'] if r['mine_share'] is None else round(r['mine_share']*100)}% of {r['provable']}"
              + (f"  ERR {r['errors']} INVAL {r['invalid']}" if r['errors'] or r['invalid'] else ""))
    print(f"state-blind '{out['trivial']}' {out['trivial_rate']:.1%}; chance on hard half {out['chance_hard']:.1%}")


if __name__ == "__main__":
    main()
