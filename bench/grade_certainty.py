"""Score the certainty probe across every answerer.

Each probe names a Minesweeper square whose status the visible numbers *prove*.
The right answer is exactly 1.0 or exactly 0.0. Three numbers per answerer:

  gap    mean P(safe | provably safe) - mean P(safe | provably a mine).
         Perfect +1.000. Saying the same thing either way scores 0.000 however
         confident that number looks.
  AUROC  probability a random provably-safe square is given a higher P(safe)
         than a random provably-mine square. Chance 0.500. This is scale-free,
         so an answerer that is merely timid - correct ordering, compressed
         range - still scores well here while scoring near zero on `gap`.
  p      Mann-Whitney U (normal approximation, tie-corrected) against chance.

`distinct` and `sd` are reported because the interesting null here is not "the
model returns a constant". These models return a wide spread of numbers; the
spread is simply uncorrelated with the answer.

    python bench/grade_certainty.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import ANSWERS, POSITIONS  # noqa: E402

ORDER = ["Haiku-4.5", "OpenJev", "Lux", "Nox", "Sol", "Eos", "Lex", "Kai", "Laya", "Jev", "CLM"]


def auroc_and_p(safe: list[float], mine: list[float]) -> tuple[float, float]:
    """AUROC via rank-sum, with the tie-corrected normal approximation of U."""
    n1, n2 = len(safe), len(mine)
    if not n1 or not n2:
        return float("nan"), 1.0
    vals = sorted([(v, 0) for v in safe] + [(v, 1) for v in mine])
    ranks, i, tie_term = {}, 0, 0.0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
            j += 1
        r = (i + j) / 2 + 1                       # average rank, 1-based
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    r1 = sum(ranks[k] for k, (_, lab) in enumerate(vals) if lab == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    auroc = u1 / (n1 * n2)
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))) if n > 1 else 0.0
    if sd == 0:
        return auroc, 1.0
    z = (u1 - n1 * n2 / 2) / sd
    p = math.erfc(abs(z) / math.sqrt(2))          # two-sided
    return auroc, p


def main():
    truth = json.loads((POSITIONS / "certainty.truth.json").read_text())
    n_safe = sum(1 for v in truth.values() if not v)
    print(f"certainty probe: {len(truth)} squares whose status the numbers prove "
          f"({n_safe} safe / {len(truth) - n_safe} mines)")
    print(f"{'answerer':14s}{'n':>5s}{'mean|safe':>11s}{'mean|mine':>11s}"
          f"{'GAP':>9s}{'AUROC':>8s}{'p':>9s}{'distinct':>10s}{'sd':>8s}")
    print("-" * 76)

    rows = []
    for f in sorted(ANSWERS.glob("certainty__*.json")):
        a = json.loads(f.read_text())
        said = {k: v for k, v in a["said"].items() if v is not None}
        safe = [v for k, v in said.items() if k in truth and not truth[k]]
        mine = [v for k, v in said.items() if k in truth and truth[k]]
        if not safe or not mine:
            continue
        gap = statistics.mean(safe) - statistics.mean(mine)
        auroc, p = auroc_and_p(safe, mine)
        rows.append({"answerer": a["answerer"], "n": len(safe) + len(mine),
                     "mean_safe": statistics.mean(safe), "mean_mine": statistics.mean(mine),
                     "gap": gap, "auroc": auroc, "p": p,
                     "distinct": len({round(v, 4) for v in said.values()}),
                     "sd": statistics.pstdev(list(said.values()))})
    rows.sort(key=lambda r: ORDER.index(r["answerer"]) if r["answerer"] in ORDER else 99)
    for r in rows:
        print(f"{r['answerer']:14s}{r['n']:5d}{r['mean_safe']:11.3f}{r['mean_mine']:11.3f}"
              f"{r['gap']:+9.3f}{r['auroc']:8.3f}{r['p']:9.3f}{r['distinct']:10d}"
              f"{r['sd']:8.3f}" + ("  *" if r["p"] < 0.05 else ""))
    print("\nperfect: gap +1.000, AUROC 1.000.   chance: gap 0.000, AUROC 0.500.")

    dest = Path(__file__).resolve().parent / "certainty.json"
    dest.write_text(json.dumps(rows, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
