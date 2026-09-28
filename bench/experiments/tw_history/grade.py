"""Grade the TextWorld history experiment and write summary.json for the report.

Arms (state only differs; candidates, instructions and truth are identical):
    A  flat text, no history  - the frozen benchmark input, re-asked in this session
    B  flat text + "ACTIONS TAKEN SO FAR: ..."
    C  Jev only: TypeSafe's recommended JSON state, no history
    D  Jev only: JSON state + actions_taken array

Every answerer's A and B come from the same session and the same code path, so
the A->B change is paired position by position (exact McNemar). Jev is not
deterministic, so it is run three times per arm and reported as the mean with
the range; its paired test uses run 1 of each arm.
"""
import json, math, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from spec import wilson  # noqa: E402

POS = {p["pid"]: p for p in json.loads((HERE / "textworld_history.json").read_text())}
LATE = {k for k, p in POS.items() if p["meta"]["step"] >= 2}
ORDER = ["Jev", "OpenJev", "CLM", "Haiku-4.5", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex", "Laya", "heuristic"]


def ok(pid, pick):
    return pick in POS[pid]["truth"]["optimal_ids"]


def mcnemar(a, b):
    n01 = sum(1 for k in a if not a[k] and b[k]); n10 = sum(1 for k in a if a[k] and not b[k])
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return n01, n10, p


def picks(name, arm, run=1):
    f = HERE / f"answers__{name}__{arm}{'' if run == 1 else f'.r{run}'}.json"
    return json.loads(f.read_text())["picks"] if f.exists() else None


def score(pk, subset):
    ks = [k for k in POS if k in subset]
    x = sum(ok(k, pk.get(k)) for k in ks)
    return {"n": len(ks), "correct": x, "rate": x / len(ks), "ci": list(wilson(x, len(ks))),
            "invalid": sum(1 for k in ks if k in pk and pk[k] not in {c["id"] for c in POS[k]["candidates"]}),
            "missing": sum(1 for k in ks if k not in pk)}


def main():
    head_dir = HERE.parent.parent / "answers"
    rows = []
    for name in ORDER:
        A, B = picks(name, "A"), picks(name, "B")
        if not (A and B):
            print(f"  (skipping {name}: arms incomplete)")
            continue
        row = {"name": name}
        for lab, sub in (("all", set(POS)), ("late", LATE)):
            a, b = score(A, sub), score(B, sub)
            n01, n10, p = mcnemar({k: ok(k, A.get(k)) for k in sub}, {k: ok(k, B.get(k)) for k in sub})
            row[lab] = {"A": a, "B": b, "fixed": n01, "broken": n10, "p": p}
        runs = {arm: [r for i in (1, 2, 3) if (r := picks(name, arm, i))] for arm in "AB"}
        if len(runs["A"]) > 1:
            row["runs"] = {arm: [score(r, set(POS))["rate"] for r in rs] for arm, rs in runs.items()}
        hf = head_dir / f"textworld__{name}.json"
        if hf.exists():
            H = json.loads(hf.read_text())["picks"]
            row["A_matches_headline"] = sum(H.get(k) == A.get(k) for k in POS) / len(POS)
            row["headline_rate"] = score(H, set(POS))["rate"]
        steps = {}
        for k, p in POS.items():
            s = min(p["meta"]["step"], 4)
            steps.setdefault(s, {"A": [], "B": []})
            steps[s]["A"].append(ok(k, A.get(k))); steps[s]["B"].append(ok(k, B.get(k)))
        row["by_step"] = {s: {arm: sum(v) / len(v) for arm, v in d.items()} | {"n": len(d["A"])}
                          for s, d in sorted(steps.items())}
        rows.append(row)
    fmt = {}
    for arm in "ABCD":
        rs = [r for i in (1, 2, 3) if (r := picks("Jev", arm, i))]

        if rs:
            fmt[arm] = {"runs": [score(r, set(POS))["rate"] for r in rs],
                        "late_runs": [score(r, LATE)["rate"] for r in rs]}
    jA, jC, jB, jD = (picks("Jev", x) for x in "ACBD")
    tests = {}
    for lab, x, y in (("A_vs_C", jA, jC), ("B_vs_D", jB, jD)):
        if x and y:
            n01, n10, p = mcnemar({k: ok(k, x[k]) for k in POS}, {k: ok(k, y[k]) for k in POS})
            tests[lab] = {"fixed": n01, "broken": n10, "p": p}
    out = {"n": len(POS), "n_late": len(LATE), "rows": rows, "jev_format": fmt, "jev_format_tests": tests,
           "example": {"pid": "tw0164", "history": POS["tw0164"]["meta"]["history"]}}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{'':12s} {'A all':>7s} {'B all':>7s} {'A late':>7s} {'B late':>7s}   fixed/broken  p        A==headline")
    for r in rows:
        print(f"{r['name']:12s} {r['all']['A']['rate']:7.1%} {r['all']['B']['rate']:7.1%} "
              f"{r['late']['A']['rate']:7.1%} {r['late']['B']['rate']:7.1%}   "
              f"+{r['all']['fixed']:3d}/-{r['all']['broken']:<3d}    {r['all']['p']:.2g}"
              f"   {r.get('A_matches_headline', float('nan')):.1%}  inval A/B {r['all']['A']['invalid']}/{r['all']['B']['invalid']}"
              + (f"   runs {r['runs']}" if 'runs' in r else ""))
    print("Jev format:", {k: [round(x, 3) for x in v["runs"]] for k, v in fmt.items()}, tests)


if __name__ == "__main__":
    main()
