"""Play every game to the end with one answerer (runs under .venv-tw).

Models are always reached over HTTP (HttpAnswerer - the same request every hosted
model gets; local models sit behind bench/serve_local.py on loopback). Baselines
that read candidate kinds (heuristic, uniform-random) run in-process.

A game ends when it is won, lost, or out of moves (engine.budget: 3x optimal,
at least 15). Recorded per game: won, moves used, optimal length, the action list,
and distance-to-goal after every move.

    JEV_BASE_URL=... python bench/experiments/tw_play/play.py --answerer http:Jev
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from engine import Game, budget, manifest  # noqa: E402
import answerers                            # noqa: E402

RESULTS = HERE / "results"


def play(a, entry):
    g = Game(entry)
    dist, errs, t0 = [g.distance()], 0, time.time()
    while not g.done and len(g.actions) < budget(entry["oracle_len"]):
        pos = g.position()
        valid = {c["id"] for c in pos["candidates"]}
        for attempt in range(3):
            try:
                choice = (g.state["policy_commands"][0] if a == "oracle" else a.pick(pos)[0])
                break
            except Exception:
                errs += 1; time.sleep(2)
        else:
            raise RuntimeError(f"{a}: game {entry['seed']} unreachable")
        if choice not in valid:
            raise RuntimeError(f"{a}: invalid choice {choice!r} in game {entry['seed']}")
        g.step(choice)
        dist.append(g.distance())
    return {"seed": entry["seed"], "won": g.won, "lost": g.lost, "moves": len(g.actions),
            "oracle_len": entry["oracle_len"], "budget": budget(entry["oracle_len"]),
            "actions": g.actions, "distance": dist, "retries": errs,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", required=True)
    ap.add_argument("--games", type=int, default=None)
    args = ap.parse_args()
    a = "oracle" if args.answerer == "oracle" else answerers.build(args.answerer)
    name = a if a == "oracle" else a.name
    games = manifest()[:args.games] if args.games else manifest()
    RESULTS.mkdir(exist_ok=True)
    out = {"answerer": name, "games": []}
    for i, e in enumerate(games):
        out["games"].append(play(a, e))
        if (i + 1) % 10 == 0:
            w = sum(r["won"] for r in out["games"])
            print(f"  [{name}] {i+1}/{len(games)} games, {w} won", flush=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    w = sum(r["won"] for r in out["games"])
    print(f"[{name}] won {w}/{len(out['games'])} -> results/{name}.json", flush=True)


if __name__ == "__main__":
    main()
