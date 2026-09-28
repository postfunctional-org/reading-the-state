"""Compare two machines' answers on the same frozen positions.

The models are run off the AMD runtime they were qualified on, so every number in
this suite carries the caveat that it is not the vendor's. A second, different
GPU is the strongest available answer to that: same frozen positions, same
weights, different silicon and a different Triton autotune. If the picks agree,
the numbers are a property of the model rather than of one card.

    python bench/crossdevice.py --other bench/answers-3090 --label "RTX 3090"

Reports, per (task, model):
  agree   share of positions where both machines picked the same candidate
  A / B   each machine's own optimal-action rate on that task
  delta   B - A, in points
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import ANSWERS, ENVS, load_positions, wilson  # noqa: E402

ORDER = ["Lux", "Nox", "Sol", "Eos", "Lex", "Kai", "Laya"]


def rate(positions, picks) -> tuple[int, float | None]:
    by = {p["pid"]: p for p in positions}
    ok = n = 0
    for pid, got in picks.items():
        p = by.get(pid)
        if not p or got not in {c["id"] for c in p["candidates"]}:
            continue
        n += 1
        ok += got in p["truth"]["optimal_ids"]
    return n, (ok / n if n else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True, help="directory of the second machine's answers")
    ap.add_argument("--label", default="other")
    ap.add_argument("--a-label", default="this")
    args = ap.parse_args()
    other = Path(args.other)
    if not other.is_dir():
        raise SystemExit(f"{other} is not a directory")

    out, worst = {}, (1.0, None)
    for env in ENVS:
        try:
            positions = load_positions(env)
        except FileNotFoundError:
            continue
        rows = []
        for f in sorted(ANSWERS.glob(f"{env}__*.json")):
            name = json.loads(f.read_text())["answerer"]
            g = other / f.name
            if not g.exists():
                continue
            a, b = json.loads(f.read_text())["picks"], json.loads(g.read_text())["picks"]
            # Only answerers actually re-run on the other machine belong here. An
            # answer file that was merely copied across would otherwise report
            # 100% agreement and read as a replication when nothing was replicated.
            shared = sorted(set(a) & set(b))
            if not shared:
                continue
            same = sum(1 for pid in shared if a[pid] == b[pid])
            agree = same / len(shared)
            na, ra = rate(positions, a)
            nb, rb = rate(positions, b)
            lo, hi = wilson(same, len(shared))
            rows.append({"model": name, "n": len(shared), "agree": agree, "agree_ci": [lo, hi],
                         "a_rate": ra, "b_rate": rb,
                         "delta": (rb - ra) if (ra is not None and rb is not None) else None,
                         "disagreements": [pid for pid in shared if a[pid] != b[pid]][:20]})
            if agree < worst[0]:
                worst = (agree, f"{name}/{env}")
        rows.sort(key=lambda r: ORDER.index(r["model"]) if r["model"] in ORDER else 99)
        if not rows:
            continue
        out[env] = rows
        print(f"\n=== {env.upper()}  ({len(positions)} positions)")
        print(f"{'model':10s}{'n':>5s}{'agree':>9s}{'95% CI':>16s}"
              f"{args.a_label:>10s}{args.label:>10s}{'delta':>9s}")
        print("-" * 69)
        for r in rows:
            lo, hi = r["agree_ci"]
            ci = f"[{lo:.0%}, {hi:.0%}]"
            print(f"{r['model']:10s}{r['n']:5d}{r['agree']:9.1%}{ci:>16s}"
                  f"{(r['a_rate'] or 0):10.1%}{(r['b_rate'] or 0):10.1%}"
                  f"{(r['delta'] or 0):+9.1%}" + ("" if r["agree"] == 1 else "  *"))

    dest = Path(__file__).resolve().parent / "crossdevice.json"
    dest.write_text(json.dumps({"label": args.label, "envs": out}, indent=1))
    print(f"\nwrote {dest}")
    if worst[1]:
        print(f"lowest agreement anywhere: {worst[0]:.1%} ({worst[1]})")
    print("* = the two machines did not pick identically on every position")


if __name__ == "__main__":
    main()
