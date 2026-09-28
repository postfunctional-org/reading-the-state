"""Does an answerer mis-address the Minesweeper grid?

Two checks, per answerer, on the frozen minesweeper set:
  1. provable landings: of its picks that land on a square the numbers prove,
     how many are mines (uniform random is the reference);
  2. transpose: on positions where swapping its pick's column and row is also a
     hidden square, how often the transposed pick would have been optimal,
     against how often the actual pick was (two-sided Fisher exact).

Written to reproduce the figures the report quotes for OpenJev and Haiku before
applying the same test to anything new.
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spec import load_positions  # noqa: E402


def fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]]"""
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def pr(x): return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def analyze(name, pos):
    picks = json.loads((ROOT / "answers" / f"minesweeper__{name}.json").read_text())["picks"]
    import re
    ntouch = lambda d: len(m.group(1).split(",")) if (m := re.search(r"touching ([\d,]+)", d)) else 0
    mines = safe = n_t = hit = hit_t = 0; rnd_t = 0.0; salient = 0; rank = []
    for p in pos:
        k = picks.get(p["pid"])
        if k is None:
            continue
        pm = p["truth"]["extra"]["p_mine"]; opt = set(p["truth"]["optimal_ids"])
        if pm.get(k) == 1.0: mines += 1
        elif pm.get(k) == 0.0: safe += 1
        c, r = k[1:].split("r"); t = f"c{r}r{c}"
        if t in pm:          # diagonal picks included, as in the published figures
            n_t += 1; hit += k in opt; hit_t += t in opt
            ids = [c["id"] for c in p["candidates"]]; rnd_t += sum(i in opt for i in ids) / len(ids)
        tc = {c["id"]: ntouch(c["desc"]) for c in p["candidates"]}
        salient += tc.get(k, -1) == max(tc.values())
        cs = [c for c in tc if c in pm]
        if len({pm[c] for c in cs}) > 1 and k in pm:
            rank.append((sum(pm[c] < pm[k] for c in cs) + (sum(pm[c] == pm[k] for c in cs) - 1) / 2) / (len(cs) - 1))
    return {"answerer": name, "provable": mines + safe, "mines": mines,
            "mine_share": mines / max(1, mines + safe), "n_transpose": n_t,
            "rate": hit / max(1, n_t), "rate_transposed": hit_t / max(1, n_t),
            "fisher_p": fisher(hit, n_t - hit, hit_t, n_t - hit_t) if n_t else None,
            "rate_random_candidate": rnd_t / max(1, n_t),
            "picks_most_touching": salient / len(pos), "danger_rank": sum(rank) / len(rank)}


if __name__ == "__main__":
    pos = load_positions("minesweeper")
    names = sys.argv[1:] or ["OpenJev", "Haiku-4.5", "Jev", "CLM", "Lux", "Nox", "uniform-random"]
    out = [analyze(n, pos) for n in names]
    u = next((r for r in out if r["answerer"] == "uniform-random"), None) or analyze("uniform-random", pos)
    for r in out:   # landing share vs uniform random's, two-sided Fisher
        r["mine_vs_uniform_p"] = fisher(r["mines"], r["provable"] - r["mines"], u["mines"], u["provable"] - u["mines"])
    for r in out:
        print(f"{r['answerer']:15s} provable {r['provable']:3d}  mines {r['mines']:3d} ({r['mine_share']:.0%})"
              f" p={r['mine_vs_uniform_p']:.2g}   transpose n={r['n_transpose']:3d}  {r['rate']:.1%} -> {r['rate_transposed']:.1%}  p={r['fisher_p']:.3g}")
    (Path(__file__).parent / "summary.json").write_text(json.dumps(out, indent=1))
