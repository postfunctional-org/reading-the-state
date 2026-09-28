"""One TextWorld game as a sequence of benchmark positions (runs under .venv-tw).

Every turn is shown to an answerer in exactly the benchmark's format, the same
as the history experiment's arm B: objective, room, inventory, then
    ACTIONS TAKEN SO FAR: 1. ...; 2. ...
where the history is the answerer's OWN actions this game. Candidates are the
game's admissible commands, shuffled with a seed derived from the state text, so
two answerers standing in the same state see the same order.
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "experiments" / "tw_history"))
from export_textworld import INSTRUCTIONS, kind_of, render_state  # noqa: E402
from spec import shuffle_candidates                                  # noqa: E402

GAMES = HERE / "games"
MANIFEST = GAMES / "manifest.json"


def history_line(prefix):
    if not prefix:
        return "ACTIONS TAKEN SO FAR: none (this is the start of the game)."
    return "ACTIONS TAKEN SO FAR: " + "; ".join(f"{i+1}. {c}" for i, c in enumerate(prefix)) + "."


def budget(oracle_len: int) -> int:
    """Moves allowed: three times the optimal length, never fewer than 15."""
    return max(15, 3 * oracle_len)


class Game:
    def __init__(self, entry: dict):
        import textworld
        from textworld import EnvInfos
        self.entry = entry
        infos = EnvInfos(admissible_commands=True, policy_commands=True, description=True,
                         inventory=True, objective=True, won=True, lost=True)
        self.env = textworld.start(str(GAMES / entry["file"]), request_infos=infos)
        self.state = self.env.reset()
        self.actions: list[str] = []
        self.done = False

    @property
    def won(self):
        return bool(self.state.get("won"))

    @property
    def lost(self):
        return bool(self.state.get("lost"))

    def distance(self):
        pol = self.state.get("policy_commands")
        return 0 if self.won else (len(pol) if pol else None)

    def position(self) -> dict:
        text = render_state(self.state) + "\n\n" + history_line(self.actions)
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        cands = shuffle_candidates([{"id": c, "desc": c, "kind": kind_of(c)}
                                    for c in (self.state.get("admissible_commands") or [])], seed=seed)
        return {"env": "textworld", "pid": f"{self.entry['seed']}:{len(self.actions)}",
                "state": text, "instructions": INSTRUCTIONS, "candidates": cands}

    def step(self, cmd: str):
        self.state, _r, done = self.env.step(cmd)
        self.actions.append(cmd)
        self.done = bool(done or self.won or self.lost)


def manifest():
    return json.loads(MANIFEST.read_text())
