"""Turn Haiku's session files into results/Haiku-4.5.json by replaying every
game's recorded moves through engine.Game - the outcome is recomputed, never
taken from the agents' own reports.  (.venv-tw)
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import Game, budget, manifest  # noqa: E402

out = {"answerer": "Haiku-4.5", "source": "workflow haiku-tw-play (wf_818dd6f1-76c): one-command interface tw_turn.sh, outcomes replayed here", "games": []}
missing = []
for e in manifest():
    f = HERE / "sessions" / f"{e['seed']}.json"
    if not f.exists():
        missing.append(e["seed"]); continue
    s = json.loads(f.read_text()); g = Game(e); dist = [g.distance()]
    for a in s["actions"]:
        g.step(a); dist.append(g.distance())
    finished = g.done or len(g.actions) >= budget(e["oracle_len"])
    out["games"].append({"seed": e["seed"], "won": g.won, "lost": g.lost, "moves": len(g.actions),
                         "oracle_len": e["oracle_len"], "budget": budget(e["oracle_len"]),
                         "actions": s["actions"], "distance": dist, "invalid_attempts": s.get("invalid", 0),
                         "finished": finished})
(HERE / "results" / "Haiku-4.5.json").write_text(json.dumps(out, indent=1))
unfinished = [g["seed"] for g in out["games"] if not g["finished"]]
print(f"Haiku: {sum(g['won'] for g in out['games'])}/{len(out['games'])} won; missing {missing}; unfinished {unfinished}")
