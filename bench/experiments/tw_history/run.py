"""Run an HTTP answerer over the TextWorld positions in one of four state arms.

    A  flat text, no history      (the frozen positions; the headline run)
    B  flat text + history line
    C  structured JSON, no history   (TypeSafe's SKILL.md: named fields for multi-part context)
    D  structured JSON + actions_taken array

Candidates, instructions and ground truth are untouched; only `state` differs.

    JEV_BASE_URL=... JEV_API_KEY=... python bench/experiments/tw_history/run.py --answerer http:Jev --arms BCD
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))


def structured(p, with_history):
    obj, room, inv = p["state"].split("\n\n")[:3]
    s = {"objective": obj.removeprefix("OBJECTIVE: "), "current_room": room, "inventory": inv}
    if with_history:
        s["actions_taken"] = p["meta"]["history"]
    return s


def arm_state(p, arm):
    flat_no_hist = p["state"].rsplit("\n\n", 1)[0]
    return {"A": flat_no_hist, "B": p["state"],
            "C": structured(p, False), "D": structured(p, True)}[arm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", required=True)
    ap.add_argument("--arms", default="BCD")
    ap.add_argument("--run", type=int, default=1, help="repeat index; >1 writes .rN files")
    args = ap.parse_args()
    import answerers
    a = answerers.build(args.answerer)
    pos = json.loads((HERE / "textworld_history.json").read_text())
    for arm in args.arms:
        picks, telem, errors, t0 = {}, {}, {}, time.time()
        for p in pos:
            q = dict(p, state=arm_state(p, arm))
            try:
                c, info = a.pick(q); picks[p["pid"]] = c; telem[p["pid"]] = info
            except Exception as e:
                errors[p["pid"]] = f"{type(e).__name__}: {str(e)[:160]}"
        tag = "" if args.run == 1 else f".r{args.run}"
        out = HERE / f"answers__{a.name}__{arm}{tag}.json"
        out.write_text(json.dumps({"answerer": a.name, "arm": arm, "picks": picks,
                                   "telemetry": telem, "errors": errors,
                                   "wall_seconds": round(time.time() - t0, 1)}, indent=1))
        print(f"[{a.name}/{arm}] {len(picks)} answered, {len(errors)} errors -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
