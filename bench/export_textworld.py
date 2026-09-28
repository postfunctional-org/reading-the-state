"""Export graded positions from real Microsoft TextWorld games.

Runs under .venv-tw (textworld pulls its own dependency stack); it only writes
JSON, so the model backends never need TextWorld installed.

Ground truth comes from TextWorld's own oracle. `EnvInfos(policy_commands=True)`
returns the winning command sequence *from the current state*, so its length is
the exact distance to the goal. Grading every candidate is then a replay: the
game is deterministic, so for each admissible command we re-run the recorded
prefix plus that command and read the new distance. Cost is the extra moves it
costs versus optimal play:

    cost(c) = distance_after(c) - (distance_now - 1)

0 for an optimal command, 1 for a wasted turn (examine, look, inventory, a drop
that gets undone), 2 for a step in the wrong direction, and so on. This is a real
value function, not a walkthrough: it grades states the walkthrough never visits.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import save_positions, save_questions, shuffle_candidates, summarise_set  # noqa: E402

INSTRUCTIONS = (
    "You are playing a text adventure. Read the situation and the list of commands "
    "available right now, then choose the single command that makes the most "
    "progress toward completing the objective."
)

# Candidate kinds, so trivial "always do X" policies can be scored.
def kind_of(cmd: str) -> str:
    head = cmd.split()[0].lower()
    if head in ("go",):
        return "go"
    if head in ("take", "pick"):
        return "take"
    if head in ("drop", "put", "insert"):
        return "put"
    if head in ("open", "close", "lock", "unlock"):
        return "manipulate"
    if head in ("examine", "look", "inventory", "read"):
        return "inspect"
    if head in ("eat", "drink"):
        return "consume"
    return "other"


def trim_objective(text: str) -> str:
    """TextWorld's objective opens with a paragraph of flavour. Keep the task."""
    t = " ".join((text or "").split())
    for marker in ("Here is how to play!", "Here is your task.", "Your task is"):
        if marker in t:
            t = t.split(marker, 1)[1].strip()
            break
    return t


def render_state(info: dict) -> str:
    desc = " ".join((info.get("description") or "").split())
    inv = " ".join((info.get("inventory") or "").split())
    obj = trim_objective(info.get("objective"))
    return (f"OBJECTIVE: {obj}\n\n{desc}\n\n{inv}").strip()


def build(n_games: int = 90, per_game: int = 3, seed0: int = 1000,
          nb_rooms: int = 8, nb_objects: int = 12, quest_length: int = 5) -> list[dict]:
    import textworld
    from textworld import EnvInfos, GameOptions
    from textworld.generator import compile_game, make_game

    infos = EnvInfos(admissible_commands=True, policy_commands=True,
                     description=True, inventory=True, objective=True,
                     won=True, lost=True)
    positions: list[dict] = []
    rng = random.Random(7)
    made = failed = 0

    for gi in range(n_games):
        # A fresh directory per game: TextWorld asserts if two games with the same
        # generated id land in one directory. And the quest generator cannot always
        # hit an exact length for a given room/object count, so try a few.
        gfile = None
        for ql in (quest_length, quest_length + 1, quest_length - 1, quest_length + 2):
            opts = GameOptions()
            opts.seeds = seed0 + gi
            opts.nb_rooms = nb_rooms
            opts.nb_objects = nb_objects
            opts.quest_length = max(2, ql)
            opts.quest_breadth = 2
            opts.path = tempfile.mkdtemp()
            try:
                gfile = compile_game(make_game(opts), opts)
                break
            except Exception:
                continue
        if gfile is None:
            failed += 1
            continue
        made += 1

        env = textworld.start(gfile, request_infos=infos)
        state = env.reset()
        prefix: list[str] = []
        taken = 0
        # walk the oracle line, sampling positions along it
        for step in range(40):
            pol = state.get("policy_commands")
            adm = state.get("admissible_commands") or []
            if not pol or len(adm) < 3:
                break
            dist_now = len(pol)
            if taken < per_game and rng.random() < 0.55:
                # env.copy() snapshots the interpreter, so each candidate costs one
                # step instead of replaying the whole prefix from the game file.
                cost = {}
                for c in adm:
                    e2 = env.copy()
                    s2, _r, _d = e2.step(c)
                    pol2 = s2.get("policy_commands")
                    if s2.get("won"):
                        d2 = 0
                    elif s2.get("lost") or not pol2:
                        d2 = None
                    else:
                        d2 = len(pol2)
                    cost[c] = None if d2 is None else max(0, d2 - (dist_now - 1))
                # the snapshots must not have disturbed the live game
                assert len(state["policy_commands"]) == dist_now, "env.copy() leaked state"
                live = {k: v for k, v in cost.items() if v is not None}
                if live and min(live.values()) == 0:
                    worst = max(live.values()) + 1
                    full = {k: (v if v is not None else worst) for k, v in cost.items()}
                    cands = shuffle_candidates(
                        [{"id": c, "desc": c, "kind": kind_of(c)} for c in adm],
                        seed=seed0 + gi * 100 + step)
                    positions.append({
                        "env": "textworld",
                        "pid": f"tw{len(positions):04d}",
                        "meta": {"game_seed": seed0 + gi, "step": step,
                                 "stratum": f"dist{min(dist_now, 6)}",
                                 "distance_now": dist_now,
                                 "generator": "textworld 1.7.0",
                                 "opts": {"nb_rooms": nb_rooms, "nb_objects": nb_objects,
                                          "quest_length": quest_length}},
                        "state": render_state(state),
                        "instructions": INSTRUCTIONS,
                        "candidates": cands,
                        "truth": {"cost": full,
                                  "optimal_ids": sorted(k for k, v in full.items() if v == 0),
                                  "scale": "moves",
                                  "extra": {"distance_now": dist_now}},
                    })
                    taken += 1
            nxt = pol[0]
            state, _r, done = env.step(nxt)
            prefix.append(nxt)
            if done or state.get("won") or state.get("lost"):
                break
    print(f"generated {made} games, {failed} failed to generate", flush=True)
    return positions


def main():
    positions = build()
    save_positions("textworld", positions)
    save_questions("textworld", positions)
    s = summarise_set("textworld", positions)
    print(json.dumps({k: v for k, v in s.items() if k != "trivial_policies"}, indent=1))
    print("trivial policies (these read nothing about the state):")
    for t in s["trivial_policies"]:
        print(f"   {t['policy']:22s} {t['rate']:6.1%}  mean cost {t['mean_cost']:.2f}")


if __name__ == "__main__":
    main()
