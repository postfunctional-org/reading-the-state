"""Per-stratum results, and the hard-subset test.

`grade.py` answers "does this answerer beat a policy that ignores the state?".
This answers the sharper question underneath it: **on the positions where that
policy is wrong, does the answerer get them right?**

Split every set in two by the best trivial policy:

    easy  - positions the best trivial policy already gets right
    hard  - positions it gets wrong

An answerer that has merely learned the set's dominant action scores high on
`easy` and at chance on `hard`. An answerer that is reading the state scores
above chance on `hard`. The `hard` column is the one to quote.

    python bench/breakdown.py
    python bench/breakdown.py --env textworld
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import (ANSWERS, ENVS, best_trivial, chance_rate, fisher_exact_two_sided,  # noqa: E402
                  grade, load_positions, wilson)

ORDER = ["oracle", "heuristic", "OpenJev", "Lux", "Nox", "Sol", "Eos", "Lex", "Kai", "Laya",
         "Haiku-4.5", "Jev", "CLM", "uniform-random"]


def rank(name: str) -> int:
    return ORDER.index(name) if name in ORDER else len(ORDER)


def split_by_trivial(positions: list[dict], bt: dict) -> tuple[list, list]:
    """easy = the best trivial policy is right here; hard = it is wrong."""
    easy, hard = [], []
    for p in positions:
        pick = next((c for c in p["candidates"] if c["kind"] == bt["kind"]), None) \
            or p["candidates"][0]
        (easy if pick["id"] in p["truth"]["optimal_ids"] else hard).append(p)
    return easy, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="all", choices=("all",) + ENVS)
    args = ap.parse_args()
    envs = ENVS if args.env == "all" else (args.env,)

    out = {}
    for env in envs:
        try:
            positions = load_positions(env)
        except FileNotFoundError:
            continue
        bt = best_trivial(positions)
        easy, hard = split_by_trivial(positions, bt)
        ch_all, ch_hard = chance_rate(positions), chance_rate(hard)
        strata = sorted({p["meta"].get("stratum") for p in positions if p["meta"].get("stratum")})

        print(f"\n=== {env.upper()}   best trivial = \"{bt['policy']}\" "
              f"({bt['rate']:.1%} overall)")
        print(f"    easy subset: {len(easy)} positions it gets right   "
              f"hard subset: {len(hard)} positions it gets wrong "
              f"(chance there {ch_hard:.1%})")
        head = (f"{'answerer':16s}{'overall':>9s}{'easy':>9s}{'HARD':>9s}"
                f"{'95% CI (hard)':>17s}{'vs chance':>11s}")
        print(head + "".join(f"{s:>11s}" for s in strata))
        print("-" * (len(head) + 11 * len(strata)))

        for f in sorted(ANSWERS.glob(f"{env}__*.json")):
            a = json.loads(f.read_text())
            picks = a["picks"]
            g, ge, gh = (grade(positions, picks), grade(easy, picks), grade(hard, picks))
            if not g["n"]:
                continue
            # is the hard-subset rate above the chance rate ON THE HARD SUBSET?
            exp = round(ch_hard * gh["n"])
            p = fisher_exact_two_sided(gh["correct"], gh["n"] - gh["correct"],
                                       exp, gh["n"] - exp) if gh["n"] else 1.0
            lo, hi = wilson(gh["correct"], gh["n"]) if gh["n"] else (0, 0)
            ci = f"[{lo:.0%}, {hi:.0%}]"
            row = (f"{a['answerer']:16s}{g['rate']:9.1%}"
                   f"{(ge['rate'] or 0):9.1%}{(gh['rate'] or 0):9.1%}{ci:>17s}"
                   f"{p:11.3f}")
            for s in strata:
                r = g["by_stratum"].get(s)
                row += f"{(r['rate'] if r else 0):11.1%}"
            print(row + ("  *" if p < 0.05 else ""))
            out.setdefault(env, {})[a["answerer"]] = {
                "overall": g["rate"], "easy": ge["rate"], "hard": gh["rate"],
                "hard_n": gh["n"], "hard_ci": [lo, hi], "hard_p_vs_chance": p,
                "chance_hard": ch_hard, "by_stratum": g["by_stratum"],
                "picked_kind": g["picked_kind"], "kind": a.get("kind", "model"),
            }
        out.setdefault(env, {})["__meta__"] = {
            "best_trivial": bt, "n_easy": len(easy), "n_hard": len(hard),
            "chance_all": ch_all, "chance_hard": ch_hard, "strata": strata}

    dest = Path(__file__).resolve().parent / "breakdown.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
