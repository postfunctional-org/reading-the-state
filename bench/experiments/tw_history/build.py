"""Side experiment: the TextWorld positions, plus the actions already taken.

Positions store only a snapshot. The export walks TextWorld's oracle line, so
the history of position (game_seed, step) is the first `step` oracle commands.
This rebuilds each game exactly as export_textworld.build() did, replays that
prefix, and refuses any position whose re-rendered state is not byte-identical
to the frozen one. Output: textworld_history.json (same format, state augmented).

    .venv-tw/bin/python bench/experiments/tw_history/build.py
"""
import json, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from export_textworld import render_state  # noqa: E402
from spec import load_positions            # noqa: E402


def history_line(prefix):
    if not prefix:
        return "ACTIONS TAKEN SO FAR: none (this is the start of the game)."
    return "ACTIONS TAKEN SO FAR: " + "; ".join(f"{i+1}. {c}" for i, c in enumerate(prefix)) + "."


def main():
    import textworld
    from textworld import EnvInfos, GameOptions
    from textworld.generator import compile_game, make_game

    pos = load_positions("textworld")
    by_game = {}
    for p in pos:
        by_game.setdefault(p["meta"]["game_seed"], []).append(p)
    infos = EnvInfos(admissible_commands=True, policy_commands=True, description=True,
                     inventory=True, objective=True, won=True, lost=True)
    out, bad = [], []
    for seed, ps in sorted(by_game.items()):
        o = ps[0]["meta"]["opts"]
        gfile = None
        for ql in (o["quest_length"], o["quest_length"] + 1, o["quest_length"] - 1, o["quest_length"] + 2):
            opts = GameOptions(); opts.seeds = seed; opts.nb_rooms = o["nb_rooms"]
            opts.nb_objects = o["nb_objects"]; opts.quest_length = max(2, ql)
            opts.quest_breadth = 2; opts.path = tempfile.mkdtemp()
            try:
                gfile = compile_game(make_game(opts), opts); break
            except Exception:
                continue
        env = textworld.start(gfile, request_infos=infos)
        state = env.reset(); prefix = []
        want = {p["meta"]["step"]: p for p in ps}
        for step in range(max(want) + 1):
            if step in want:
                p = want[step]
                if render_state(state) != p["state"] or len(state["policy_commands"]) != p["meta"]["distance_now"]:
                    bad.append(p["pid"])
                else:
                    q = dict(p); q["state"] = p["state"] + "\n\n" + history_line(prefix)
                    q["meta"] = dict(p["meta"], history=list(prefix))
                    out.append(q)
            nxt = state["policy_commands"][0]
            state, _r, _d = env.step(nxt); prefix.append(nxt)
    out.sort(key=lambda p: p["pid"])
    (HERE / "textworld_history.json").write_text(json.dumps(out, indent=1))
    print(f"rebuilt {len(out)}/{len(pos)} positions, {len(bad)} mismatched: {bad[:10]}")


if __name__ == "__main__":
    main()
