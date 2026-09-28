"""The benchmark format, the baselines, and the grading. One place, so every
environment and every answerer is measured the same way.

WHY THIS FILE EXISTS
--------------------
An earlier version of this benchmark reported that a model scored 76% on a set of
"forced" roguelike positions and called it evidence of good judgement. It was not:
92% of that set was the same underlying rule, so a policy of "attack whatever is
next to you", which reads nothing else about the state, scored 92% on it. The
model was *below* a one-line heuristic while looking well above uniform random.

Two rules follow from that mistake, and both are enforced here:

1. **Every position set is stratified.** Where positions come in kinds, the set
   holds roughly equal numbers of each kind, so no single hard-coded rule can win.
2. **Every task reports its best trivial policy.** A trivial policy ignores the
   state completely and always prefers one kind of action. `best_trivial` is the
   strongest such policy on that exact set. It, not uniform random, is the bar an
   answerer has to clear before any claim about reading the state is warranted.

POSITION FORMAT
---------------
Every position, in every environment, is one JSON object:

    {
      "env":   "minesweeper" | "textworld" | "roguelike",
      "pid":   stable unique id,
      "meta":  {...}                       provenance: seed, config, rule, ...
      "state": "..."                       exactly the text an answerer is shown
      "instructions": "..."                the Choice question's instructions
      "candidates": [                      the runtime option set, already shuffled
         {"id": "...", "desc": "...", "kind": "..."}
      ],
      "truth": {
         "cost":        {id: float},       >= 0, lower is better, 0 == optimal
         "optimal_ids": [...],             every id with cost 0
         "scale":       "moves" | "probability",
         "extra":       {...}              anything environment-specific
      }
    }

`cost` is the whole grading contract. It is an exact quantity computed by a solver
or an oracle, never a proxy, and it is never shown to the answerer.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSITIONS = ROOT / "bench" / "positions"
ANSWERS = ROOT / "bench" / "answers"
ENVS = ("minesweeper", "textworld", "roguelike")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    from math import comb

    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, c1 = a + b, a + c

    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    obs = pr(a)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= obs * (1 + 1e-9)))


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def chance_rate(positions: list[dict]) -> float:
    """Exact expected accuracy of choosing uniformly among the candidates."""
    if not positions:
        return 0.0
    return statistics.mean(
        len(p["truth"]["optimal_ids"]) / len(p["candidates"]) for p in positions
    )


def chance_cost(positions: list[dict]) -> float:
    """Exact expected cost of choosing uniformly."""
    vals = []
    for p in positions:
        cs = [p["truth"]["cost"][c["id"]] for c in p["candidates"]]
        vals.append(statistics.mean(cs))
    return statistics.mean(vals) if vals else 0.0


def kinds_of(positions: list[dict]) -> list[str]:
    ks = Counter()
    for p in positions:
        for c in p["candidates"]:
            ks[c["kind"]] += 1
    return [k for k, _ in ks.most_common()]


def trivial_policies(positions: list[dict], seed: int = 0) -> list[dict]:
    """Score every 'always prefer kind K' policy. These read nothing about the state.

    Tie-break inside a kind is the first candidate in the (already shuffled)
    candidate list, so a policy cannot smuggle in extra information through
    ordering. A policy that finds no candidate of its kind falls back to the
    first candidate.
    """
    out = []
    for kind in kinds_of(positions):
        n = ok = 0
        costs = []
        for p in positions:
            pick = next((c for c in p["candidates"] if c["kind"] == kind), None)
            if pick is None:
                pick = p["candidates"][0]
            n += 1
            ok += pick["id"] in p["truth"]["optimal_ids"]
            costs.append(p["truth"]["cost"][pick["id"]])
        lo, hi = wilson(ok, n)
        out.append({"policy": f"always {kind}", "kind": kind, "n": n, "correct": ok,
                    "rate": ok / n, "ci": [lo, hi], "mean_cost": statistics.mean(costs)})
    out.sort(key=lambda r: -r["rate"])
    return out


def best_trivial(positions: list[dict]) -> dict | None:
    pol = trivial_policies(positions)
    return pol[0] if pol else None


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


def grade(positions: list[dict], picks: dict[str, str]) -> dict:
    """Score one answerer's picks against the exact costs.

    `picks` maps pid -> chosen candidate id. Unanswered positions and picks that
    are not in that position's candidate list are counted and excluded, never
    silently treated as wrong.
    """
    by_pid = {p["pid"]: p for p in positions}
    n = ok = invalid = missing = 0
    costs, by_kind, by_meta = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for pid, p in by_pid.items():
        got = picks.get(pid)
        if got is None:
            missing += 1
            continue
        ids = {c["id"]: c for c in p["candidates"]}
        if got not in ids:
            invalid += 1
            continue
        n += 1
        hit = got in p["truth"]["optimal_ids"]
        ok += hit
        costs.append(p["truth"]["cost"][got])
        k = ids[got]["kind"]
        by_kind[k][1] += 1
        by_kind[k][0] += hit
        strat = p["meta"].get("stratum")
        if strat:
            by_meta[strat][1] += 1
            by_meta[strat][0] += hit
    lo, hi = wilson(ok, n)
    return {
        "n": n, "correct": ok, "missing": missing, "invalid": invalid,
        "rate": (ok / n) if n else None, "ci": [lo, hi],
        "mean_cost": statistics.mean(costs) if costs else None,
        "picked_kind": {k: v[1] / n for k, v in by_kind.items()} if n else {},
        "by_stratum": {k: {"n": v[1], "correct": v[0], "rate": v[0] / v[1]}
                       for k, v in by_meta.items() if v[1]},
    }


def compare_to_baseline(result: dict, baseline: dict) -> dict:
    """Fisher exact of an answerer against a baseline policy on the same set."""
    if not result.get("n") or not baseline:
        return {}
    p = fisher_exact_two_sided(result["correct"], result["n"] - result["correct"],
                               baseline["correct"], baseline["n"] - baseline["correct"])
    return {"vs": baseline["policy"], "delta": result["rate"] - baseline["rate"], "p": p}


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def save_positions(env: str, positions: list[dict]) -> Path:
    POSITIONS.mkdir(parents=True, exist_ok=True)
    path = POSITIONS / f"{env}.json"
    path.write_text(json.dumps(positions, indent=1))
    return path


def load_positions(env: str) -> list[dict]:
    return json.loads((POSITIONS / f"{env}.json").read_text())


def questions_only(positions: list[dict]) -> list[dict]:
    """The answerer-visible half: no costs, no optimal ids, no provenance."""
    return [{"pid": p["pid"], "state": p["state"], "instructions": p["instructions"],
             "candidates": [{"id": c["id"], "desc": c["desc"]} for c in p["candidates"]]}
            for p in positions]


def save_questions(env: str, positions: list[dict]) -> Path:
    POSITIONS.mkdir(parents=True, exist_ok=True)
    path = POSITIONS / f"{env}.questions.json"
    path.write_text(json.dumps(questions_only(positions), indent=1))
    return path


def shuffle_candidates(cands: list[dict], seed: int) -> list[dict]:
    """Candidate order is randomised per position, always.

    A fixed order correlates slot position with action type, and these models
    have a measurable pull toward the top of the list; in an earlier run that
    confound was worth up to 64 points of accuracy.
    """
    out = list(cands)
    random.Random(seed).shuffle(out)
    return out


def summarise_set(env: str, positions: list[dict]) -> dict:
    strata = Counter(p["meta"].get("stratum", "-") for p in positions)
    return {
        "env": env,
        "n_positions": len(positions),
        "mean_candidates": statistics.mean(len(p["candidates"]) for p in positions),
        "strata": dict(strata),
        "chance_rate": chance_rate(positions),
        "chance_cost": chance_cost(positions),
        "trivial_policies": trivial_policies(positions),
        "scale": positions[0]["truth"]["scale"] if positions else None,
    }
