"""Join answers to positions and produce the comparison table.

Every answerer is compared against two bars:

  chance        - exact expected accuracy of picking uniformly among candidates
  best trivial  - the strongest policy that ignores the state entirely

`best trivial` is the one that matters. Beating chance is easy and means little;
beating the best hard-coded rule on a stratified set is the minimum evidence that
an answerer is reading anything.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import (ANSWERS, ENVS, best_trivial, chance_cost, chance_rate,  # noqa: E402
                  compare_to_baseline, grade, load_positions, summarise_set)

ORDER = ["oracle", "heuristic", "OpenJev", "Lux", "Nox", "Sol", "Eos", "Lex", "Kai", "Laya",
         "Haiku-4.5", "Jev", "CLM", "uniform-random"]


def rank(name: str) -> int:
    return ORDER.index(name) if name in ORDER else len(ORDER)


def main():
    report = {"envs": {}, "answerers": {}}
    for env in ENVS:
        try:
            positions = load_positions(env)
        except FileNotFoundError:
            continue
        if not positions:
            continue
        summary = summarise_set(env, positions)
        bt = best_trivial(positions)
        report["envs"][env] = summary

        rows = []
        for f in sorted(ANSWERS.glob(f"{env}__*.json")):
            a = json.loads(f.read_text())
            g = grade(positions, a["picks"])
            if not g["n"]:
                continue
            g["answerer"] = a["answerer"]
            g["kind"] = a.get("kind", "model")
            g["vs_trivial"] = compare_to_baseline(g, bt)
            g["vs_chance_delta"] = g["rate"] - summary["chance_rate"]
            lat = [t.get("latency_ms") for t in (a.get("telemetry") or {}).values()
                   if t and t.get("latency_ms")]
            g["mean_latency_ms"] = statistics.mean(lat) if lat else None
            rows.append(g)
            report["answerers"].setdefault(a["answerer"], {})[env] = g
        rows.sort(key=lambda r: (rank(r["answerer"]), -(r["rate"] or 0)))

        scale = summary["scale"]
        print(f"\n=== {env.upper()}  ({summary['n_positions']} positions, "
              f"strata {summary['strata']}, cost in {scale}) " + "=" * 8)
        print(f"{'answerer':16s}{'optimal':>9s}{'95% CI':>16s}{'mean cost':>11s}"
              f"{'vs trivial':>12s}{'p':>8s}{'miss':>6s}{'inval':>6s}")
        print(f"{'chance (uniform)':16s}{summary['chance_rate']:9.1%}{'':>16s}"
              f"{summary['chance_cost']:11.3f}")
        if bt:
            lo, hi = bt["ci"]
            ci = f"[{lo:.0%}, {hi:.0%}]"
            print(f"{'best trivial':16s}{bt['rate']:9.1%}{ci:>16s}"
                  f"{bt['mean_cost']:11.3f}   ({bt['policy']})")
        print("-" * 72)
        for r in rows:
            v = r["vs_trivial"]
            lo, hi = r["ci"]
            ci = f"[{lo:.0%}, {hi:.0%}]"
            cost = r["mean_cost"] if r["mean_cost"] is not None else 0.0
            print(f"{r['answerer']:16s}{r['rate']:9.1%}{ci:>16s}{cost:11.3f}"
                  f"{(v.get('delta') or 0):+12.1%}{(v.get('p') or 1):8.3f}"
                  f"{r['missing']:6d}{r['invalid']:6d}"
                  + ("  *" if (v.get("p") or 1) < 0.05 else ""))

        # A gap concentrated in one stratum unbalances the set for that answerer
        # only, and silently moves its score. This is how an earlier Haiku
        # roguelike number came out flattering: two failed batches were both
        # heal_or_die, the stratum it was worst at.
        for r in rows:
            if not (r["missing"] or r["invalid"]):
                continue
            graded = {k: v["n"] for k, v in r["by_stratum"].items()}
            built = summary["strata"]
            skew = [f"{k}: {graded.get(k, 0)}/{n}" for k, n in built.items()
                    if graded.get(k, 0) < n]
            print(f"  ! {r['answerer']}: {r['missing']} missing, {r['invalid']} invalid"
                  f"  -> graded per stratum {', '.join(skew)}"
                  f"{'   SET IS UNBALANCED FOR THIS ANSWERER' if len(skew) < len(built) else ''}")

    out = Path(__file__).resolve().parent / "report.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {out}")

    # headline: who clears the best trivial policy anywhere?
    print("\nAnswerers that beat the best state-blind policy (p<0.05):")
    any_hit = False
    for name, envs in report["answerers"].items():
        hits = [e for e, g in envs.items()
                if g["kind"] == "model" and (g["vs_trivial"].get("p") or 1) < 0.05
                and (g["vs_trivial"].get("delta") or 0) > 0]
        if hits:
            any_hit = True
            print(f"   {name:14s} {', '.join(hits)}")
    if not any_hit:
        print("   none")


if __name__ == "__main__":
    main()
