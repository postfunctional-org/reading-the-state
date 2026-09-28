"""The certainty probe: the tightest calibration target in the suite.

Minesweeper positions contain squares whose status the revealed numbers *prove* -
each is provably safe or provably a mine. For those, the correct answer to "is
this square safe?" is exactly 1.0 or exactly 0.0. No estimate, no risk appetite,
no tie-break.

So this asks a Noul question about them and scores the gap:

    gap = mean P(safe | provably safe) - mean P(safe | provably a mine)

A perfect answerer scores +1.000. Chance scores 0.000. A model that returns the
same number either way scores 0.000 however confident that number looks, which is
the failure this probe exists to catch.

    python bench/run_certainty.py --answerer Nox
    JEV_BASE_URL=... JEV_API_KEY=... python bench/run_certainty.py --answerer http:JEV
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import ANSWERS, POSITIONS, load_positions  # noqa: E402

N_PROBES = 240
PER_POSITION = 3


def noul_question(col: str, row: str) -> dict:
    return {"safe": {"type": "noul",
                     "instructions": f"The hidden square at column {col}, row {row} "
                                     f"does NOT contain a mine."}}


FROZEN = POSITIONS / "certainty.probes.json"
TRUTH = POSITIONS / "certainty.truth.json"


def build_probes(seed: int = 11, rebuild: bool = False) -> list[dict]:
    """Load the frozen probe set, or draw a fresh one with --rebuild.

    The probes are frozen on disk for the same reason the positions are: every
    answerer must be scored on byte-identical inputs. The pair of files is
    self-contained - `certainty.probes.json` is everything an answerer is shown,
    `certainty.truth.json` is the answer - so rebuilding the minesweeper set
    cannot silently redraw the probes underneath results already on disk.

    A fresh draw is balanced: half the probes are provably safe squares and half
    are provably mines, so the two means the gap is built from carry equal weight.
    """
    if FROZEN.exists() and TRUTH.exists() and not rebuild:
        truth = json.loads(TRUTH.read_text())
        return [{**pr, "is_mine": truth[pr["qid"]]} for pr in json.loads(FROZEN.read_text())]

    rng = random.Random(seed)
    safe, mine = [], []
    for p in load_positions("minesweeper"):
        certain = list(p["truth"]["extra"]["certain"])
        rng.shuffle(certain)
        for c in certain[:PER_POSITION]:
            col, row = c["id"][1:].split("r")
            probe = {"pid": p["pid"], "state": p["state"], "square": c["id"],
                     "col": col, "row": row, "is_mine": c["is_mine"]}
            (mine if c["is_mine"] else safe).append(probe)
    rng.shuffle(safe)
    rng.shuffle(mine)
    half = min(N_PROBES // 2, len(safe), len(mine))
    probes = safe[:half] + mine[:half]
    rng.shuffle(probes)
    for i, pr in enumerate(probes):
        pr["qid"] = f"cert{i:04d}"
    return probes


def ask(answerer, probe: dict) -> float | None:
    """Ask one Noul question through whichever answerer this is."""
    qs = noul_question(probe["col"], probe["row"])
    be = getattr(answerer, "be", None)
    if be is not None:                                   # Decision 1.0
        r = be.evaluate({"model": be.model_id, "state": probe["state"], "questions": qs})
        return float(r["answers"]["safe"]["noul"])
    router = getattr(answerer, "router", None)
    if router is not None:                               # Laya
        try:
            r = router.predict(probe["state"], qs, max_len=answerer.max_len)
        except TypeError:
            r = router.predict(probe["state"], qs)
        return float(r["answers"]["safe"]["noul"])
    if hasattr(answerer, "base_url"):                    # hosted SystemOne endpoint
        import urllib.request
        body = json.dumps({"model": answerer.model, "state": probe["state"],
                           "questions": qs}).encode()
        req = urllib.request.Request(
            answerer.base_url + answerer.path, data=body, method="POST",
            headers={"Authorization": f"Bearer {answerer.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=answerer.timeout) as resp:
            return float(json.loads(resp.read().decode())["answers"]["safe"]["noul"])
    raise SystemExit(f"{answerer.name} does not expose a Noul path; "
                     f"extend ask() in run_certainty.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true",
                    help="redraw the probe list (invalidates every answer on disk)")
    args = ap.parse_args()

    import answerers

    a = answerers.build(args.answerer)
    probes = build_probes(rebuild=args.rebuild)
    if args.rebuild:
        FROZEN.write_text(json.dumps(
            [{k: v for k, v in p.items() if k != "is_mine"} for p in probes], indent=1))
        TRUTH.write_text(json.dumps({p["qid"]: p["is_mine"] for p in probes}, indent=1))
        n_mine = sum(p["is_mine"] for p in probes)
        print(f"rebuilt {FROZEN.name}: {len(probes)} probes, "
              f"{len(probes)-n_mine} provably safe / {n_mine} provably mines")
    if args.limit:                       # --limit is a smoke test, never a rebuild size
        probes = probes[:args.limit]

    said, errors = {}, {}
    t0 = time.time()
    for i, p in enumerate(probes):
        try:
            said[p["qid"]] = ask(a, p)
        except Exception as e:
            errors[p["qid"]] = f"{type(e).__name__}: {str(e)[:140]}"
        if (i + 1) % 50 == 0:
            print(f"  [{a.name}] {i+1}/{len(probes)}  {time.time()-t0:.0f}s", flush=True)

    safe = [said[p["qid"]] for p in probes if not p["is_mine"] and said.get(p["qid"]) is not None]
    mine = [said[p["qid"]] for p in probes if p["is_mine"] and said.get(p["qid"]) is not None]
    gap = (statistics.mean(safe) - statistics.mean(mine)) if safe and mine else None
    res = {"answerer": a.name, "n": len(safe) + len(mine),
           "mean_when_provably_safe": statistics.mean(safe) if safe else None,
           "mean_when_provably_mine": statistics.mean(mine) if mine else None,
           "gap": gap,
           "distinct_values": len({round(v, 4) for v in said.values() if v is not None}),
           "sd": statistics.pstdev([v for v in said.values() if v is not None]) if said else None,
           "errors": errors, "said": said,
           "wall_seconds": round(time.time() - t0, 1)}

    ANSWERS.mkdir(parents=True, exist_ok=True)
    out = ANSWERS / f"certainty__{a.name.replace('/', '_')}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"[{a.name}] certainty n={res['n']}  "
          f"safe {res['mean_when_provably_safe']:.3f}  mine {res['mean_when_provably_mine']:.3f}  "
          f"GAP {gap:+.3f}   (perfect +1.000, chance 0.000)")
    print(f"   distinct values {res['distinct_values']}, sd {res['sd']:.4f} -> {out.name}")


if __name__ == "__main__":
    sys.exit(main())
