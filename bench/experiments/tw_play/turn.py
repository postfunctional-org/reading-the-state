"""Haiku's only interface to a playthrough: one game, one move at a time.

    scripts/tw_turn.sh <seed>              show the current turn
    scripts/tw_turn.sh <seed> "<command>"  play a command, then show the next turn

State lives in sessions/<seed>.json as the list of moves so far; each call replays
the game from the start, which is deterministic, so the turn shown is exactly the
position engine.Game produces for every other answerer. A command that is not one
of the current candidates is rejected and does not use a move.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import Game, budget, manifest  # noqa: E402

SESS = HERE / "sessions"


def show(g, entry, note=""):
    if note:
        print(note + "\n")
    if g.done or len(g.actions) >= budget(entry["oracle_len"]):
        res = "WON" if g.won else ("LOST" if g.lost else "OUT OF MOVES")
        print(f"GAME OVER: {res} after {len(g.actions)} moves. Nothing more to do for this game.")
        return
    p = g.position()
    print(f"TURN {len(g.actions)+1} of at most {budget(entry['oracle_len'])}\n")
    print(p["state"] + "\n")
    print("INSTRUCTIONS: " + p["instructions"] + "\n")
    print("CANDIDATES (reply with one exactly as written):")
    for c in p["candidates"]:
        print("  " + c["id"])


def main():
    seed = int(sys.argv[1]); cmd = sys.argv[2] if len(sys.argv) > 2 else None
    entry = next(e for e in manifest() if e["seed"] == seed)
    SESS.mkdir(exist_ok=True); f = SESS / f"{seed}.json"
    s = json.loads(f.read_text()) if f.exists() else {"actions": [], "invalid": 0}
    g = Game(entry)
    for a in s["actions"]:
        g.step(a)
    note = ""
    if cmd is not None:
        over = g.done or len(g.actions) >= budget(entry["oracle_len"])
        valid = {c["id"] for c in g.position()["candidates"]}
        if over:
            note = "The game is already over."
        elif cmd not in valid:
            s["invalid"] += 1
            note = f"REJECTED: {cmd!r} is not one of the candidates. No move was used. Choose again."
        else:
            g.step(cmd); s["actions"].append(cmd)
        f.write_text(json.dumps(s))
    show(g, entry, note)


if __name__ == "__main__":
    main()
