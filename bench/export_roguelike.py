"""Export graded roguelike positions - stratified, which is the fix for the flaw
that invalidated the previous version of this task.

The old set was harvested by just recording whatever forced positions turned up
during play. 92% of them were the same rule ("finish off the monster hitting
you"), so "always attack" scored 92% while reading nothing, and every model sat
below that while appearing far above uniform random.

Here the set holds an equal number of each rule. With three rules in equal parts,
an always-attack policy can reach at most about a third, and `spec.trivial_policies`
prints exactly what each such policy does score on the set that was actually built.

A forced position is one where a competent player would call every other move a
blunder, so the cost is binary: 0 for the required action, 1 for anything else.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "harness"))

from spec import save_positions, save_questions, shuffle_candidates, summarise_set  # noqa: E402

PER_RULE = 45


def build(per_rule: int = PER_RULE, max_seeds: int = 1500) -> list[dict]:
    from agents import HeuristicAgent, RandomAgent
    from rogue import Game
    from roguelike_truth import INSTRUCTIONS, tactical_ground_truth

    buckets: dict[str, list] = defaultdict(list)
    for seed in range(max_seeds):
        if all(len(v) >= per_rule for v in buckets.values()) and len(buckets) >= 3:
            break
        for AG in (HeuristicAgent, RandomAgent):
            g, a = Game(seed), AG(seed)
            while not g.done:
                key, rule = tactical_ground_truth(g)
                acts = g.legal_actions()
                if key is not None and len(acts) >= 3 and len(buckets[rule]) < per_rule:
                    cands = shuffle_candidates(
                        [{"id": x.key, "desc": x.desc, "kind": x.kind} for x in acts],
                        seed=seed * 131 + g.turn)
                    buckets[rule].append({
                        "env": "roguelike",
                        "pid": "",                     # assigned after balancing
                        "meta": {"seed": seed, "turn": g.turn, "stratum": rule,
                                 "agent": AG.name, "rule": rule},
                        "state": g.state_text(),
                        "instructions": INSTRUCTIONS,
                        "candidates": cands,
                        "truth": {
                            "cost": {x["id"]: (0 if x["id"] == key else 1) for x in cands},
                            "optimal_ids": [key],
                            "scale": "blunder",
                            "extra": {"rule": rule},
                        },
                    })
                g.step(a.act(g))

    n = min(len(v) for v in buckets.values())
    out = []
    for rule in sorted(buckets):
        for p in buckets[rule][:n]:
            p["pid"] = f"rg{len(out):04d}"
            out.append(p)
    return out


def main():
    positions = build()
    save_positions("roguelike", positions)
    save_questions("roguelike", positions)
    s = summarise_set("roguelike", positions)
    print(json.dumps({k: v for k, v in s.items() if k != "trivial_policies"}, indent=1))
    print("trivial policies (these read nothing about the state):")
    for t in s["trivial_policies"]:
        print(f"   {t['policy']:22s} {t['rate']:6.1%}")


if __name__ == "__main__":
    main()
