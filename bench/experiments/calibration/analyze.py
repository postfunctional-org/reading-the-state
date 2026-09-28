"""Choice calibration from answers already on disk: is a decision model's stated
confidence in its pick worth anything?

For every answerer that returns a probability distribution (all decision models;
Haiku returns a pick only), per task:
  conf      the probability it put on the candidate it picked
  correct   the pick is one of the optimal candidates
  ECE       expected calibration error, 10 equal-width bins, weighted by count
  over      mean confidence minus accuracy (positive = overconfident)
  brier     mean (conf - correct)^2
  auroc     P(conf on a right pick > conf on a wrong pick), ties half - does
            confidence separate its right answers from its wrong ones? (Mann-Whitney)

Jev is its first run (the published one), like every table on the page.

    python3 bench/experiments/calibration/analyze.py
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from spec import ENVS, load_positions  # noqa: E402

MODELS = ["Jev", "OpenJev", "CLM", "Laya", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex"]


def mw_p(pos, neg):
    """two-sided normal-approximation Mann-Whitney p with tie correction"""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i, ties = {}, 0, 0.0
    r = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2 + 1
        t = j - i + 1; ties += t ** 3 - t; i = j + 1
    n1, n2 = len(pos), len(neg); n = n1 + n2
    u = sum(rk for rk, (_, lab) in zip(r, allv) if lab) - n1 * (n1 + 1) / 2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    if sd == 0:
        return 1.0
    z = (u - n1 * n2 / 2) / sd
    return math.erfc(abs(z) / math.sqrt(2))


def main():
    out = {"bins": 10, "rows": []}
    for m in MODELS:
        row = {"name": m}
        for e in ENVS:
            P = {p["pid"]: p for p in load_positions(e)}
            d = json.loads((BENCH / "answers" / f"{e}__{m}.json").read_text())
            conf, ok = [], []
            for k, pick in d["picks"].items():
                probs = (d["telemetry"].get(k) or {}).get("probs") or {}
                if pick in probs:
                    conf.append(float(probs[pick])); ok.append(pick in P[k]["truth"]["optimal_ids"])
            n = len(conf)
            ece, rel = 0.0, []
            for b in range(10):
                idx = [i for i, c in enumerate(conf) if (b / 10 < c <= (b + 1) / 10) or (b == 0 and c == 0)]
                if idx:
                    mc = sum(conf[i] for i in idx) / len(idx); acc = sum(ok[i] for i in idx) / len(idx)
                    ece += len(idx) / n * abs(mc - acc); rel.append([round(mc, 4), round(acc, 4), len(idx)])
            pos = [c for c, o in zip(conf, ok) if o]; neg = [c for c, o in zip(conf, ok) if not o]
            au = (sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))) if pos and neg else None
            row[e] = {"n": n, "acc": sum(ok) / n, "mean_conf": sum(conf) / n, "over": sum(conf) / n - sum(ok) / n,
                      "ece": ece, "brier": sum((c - o) ** 2 for c, o in zip(conf, ok)) / n,
                      "auroc": au, "auroc_p": mw_p(pos, neg) if pos and neg else None, "reliability": rel}
        out["rows"].append(row)
    (HERE / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{'':9s}" + "".join(f"{e:>36s}" for e in ENVS))
    print(f"{'':9s}" + "   acc  conf  over   ECE  AUROC     p" * 3)
    for r in out["rows"]:
        s = f"{r['name']:9s}"
        for e in ENVS:
            x = r[e]
            s += (f"  {x['acc']:4.0%} {x['mean_conf']:5.2f} {x['over']:+5.2f} {x['ece']:5.3f} "
                  f"{x['auroc']:6.3f} {x['auroc_p']:5.3f}")
        print(s)


if __name__ == "__main__":
    main()
