"""Export graded Minesweeper positions, stratified by whether the position is
deducible or a genuine guess.

Ground truth is exact: enumerate every mine layout consistent with the revealed
numbers and the total mine count, and read off P(mine) per square. Cost is the
excess mine probability accepted versus the best available square, so 0 is optimal
and the scale is real probability.

Candidate `kind` is geometric - corner / edge / interior - rather than semantic,
because every candidate here is the same action ("open this square"). That still
gives `spec.trivial_policies` something to test: a policy that always opens a
corner reads the board's shape but none of its numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "harness"))

from spec import save_positions, save_questions, shuffle_candidates, summarise_set  # noqa: E402

PER_STRATUM = 135   # 135 deducible + 135 guess; 45 each was too few to
                    # call the hard-subset null with any power (see bench/breakdown.py)


def geom_kind(x: int, y: int, w: int, h: int) -> str:
    ex = x in (0, w - 1)
    ey = y in (0, h - 1)
    if ex and ey:
        return "corner"
    if ex or ey:
        return "edge"
    return "interior"


def build(per_stratum: int = PER_STRATUM, max_seeds: int = 3000) -> list[dict]:
    from mines_env import INSTRUCTIONS, MinesEpisode, cell_name, describe

    buckets: dict[str, list] = defaultdict(list)
    configs = [(9, 9, 10), (8, 8, 16)]
    for seed in range(max_seeds):
        if all(len(buckets[k]) >= per_stratum
               for k in ("deducible", "guess")) and len(buckets) >= 2:
            break
        for (w, h, m) in configs:
            ep = MinesEpisode(seed, w, h, m)
            rng = random.Random(seed * 37 + w)
            for _ in range(30):
                if ep.done:
                    break
                sol, cands = ep.solution, ep.candidates()
                if len(cands) < 3:
                    break
                truth = ep.truth()
                stratum = "deducible" if truth["deducible"] else "guess"
                if len(buckets[stratum]) < per_stratum and rng.random() < 0.3:
                    probs = {cell_name(c): sol.prob[c] for c in cands}
                    best = min(probs.values())
                    shown = shuffle_candidates(
                        [{"id": cell_name(c), "desc": describe(ep.board, c),
                          "kind": geom_kind(c[0], c[1], w, h)} for c in cands],
                        seed=seed * 101 + ep.board.moves)
                    buckets[stratum].append({
                        "env": "minesweeper",
                        "pid": "",
                        "meta": {"seed": seed, "config": f"{w}x{h}/{m}",
                                 "move": ep.board.moves, "stratum": stratum},
                        "state": ep.state_text(),
                        "instructions": INSTRUCTIONS,
                        "candidates": shown,
                        "truth": {
                            "cost": {k: v - best for k, v in probs.items()},
                            "optimal_ids": sorted(k for k, v in probs.items()
                                                  if v <= best + 1e-12),
                            "scale": "probability",
                            "extra": {"p_mine": probs, "best_p": best,
                                      "certain": [{"id": k, "is_mine": v > 0.5}
                                                  for k, v in probs.items()
                                                  if v < 1e-9 or v > 1 - 1e-9]},
                        },
                    })
                ep.step(min(cands, key=lambda c: sol.prob[c]))

    n = min(len(buckets[k]) for k in buckets)
    out = []
    for k in sorted(buckets):
        for p in buckets[k][:n]:
            p["pid"] = f"ms{len(out):04d}"
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    a = ap.parse_args()
    positions = build(a.per_stratum)
    save_positions("minesweeper", positions)
    save_questions("minesweeper", positions)
    s = summarise_set("minesweeper", positions)
    print(json.dumps({k: v for k, v in s.items() if k != "trivial_policies"}, indent=1))
    print("trivial policies (geometry only, no numbers read):")
    for t in s["trivial_policies"]:
        print(f"   {t['policy']:22s} {t['rate']:6.1%}  mean excess P(mine) {t['mean_cost']:.3f}")
    n_cert = sum(len(p["truth"]["extra"]["certain"]) for p in positions)
    print(f"\nsquares whose status the numbers PROVE, available for the certainty probe: {n_cert}")


if __name__ == "__main__":
    main()
