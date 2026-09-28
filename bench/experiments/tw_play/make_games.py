"""Compile the playthrough games once, with export_textworld's exact generator
settings, so every answerer plays byte-identical game files.

    .venv-tw/bin/python bench/experiments/tw_play/make_games.py 60
"""
import hashlib, json, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import GAMES, MANIFEST  # noqa: E402


def main(n: int, seed0: int = 1000, nb_rooms: int = 8, nb_objects: int = 12, quest_length: int = 5):
    import textworld
    from textworld import EnvInfos, GameOptions
    from textworld.generator import compile_game, make_game
    GAMES.mkdir(parents=True, exist_ok=True)
    out = []
    for gi in range(n):
        gfile = None
        for ql in (quest_length, quest_length + 1, quest_length - 1, quest_length + 2):
            opts = GameOptions(); opts.seeds = seed0 + gi; opts.nb_rooms = nb_rooms
            opts.nb_objects = nb_objects; opts.quest_length = max(2, ql); opts.quest_breadth = 2
            opts.path = tempfile.mkdtemp()
            try:
                gfile = compile_game(make_game(opts), opts); break
            except Exception:
                continue
        if gfile is None:
            print(f"  seed {seed0+gi}: failed to generate"); continue
        src = Path(gfile); dst = GAMES / f"g{seed0+gi}"
        dst.mkdir(exist_ok=True)
        for f in src.parent.glob(src.stem + ".*"):
            shutil.copy(f, dst / f.name)
        env = textworld.start(str(dst / src.name), request_infos=EnvInfos(policy_commands=True))
        olen = len(env.reset()["policy_commands"])
        out.append({"seed": seed0 + gi, "file": f"g{seed0+gi}/{src.name}", "oracle_len": olen,
                    "sha256": hashlib.sha256((dst / src.name).read_bytes()).hexdigest()})
    MANIFEST.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} games, oracle length {min(o['oracle_len'] for o in out)}-{max(o['oracle_len'] for o in out)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
