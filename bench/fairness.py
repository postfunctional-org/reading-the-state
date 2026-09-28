"""Is every answerer tested on the same footing? Exit non-zero if not.

For every answerer x every test, checks:
  coverage   every position answered, nothing invalid (or, for certainty, every probe
             answered with a number in [0, 1])
  inputs     the frozen position files and the history file are unchanged since the
             answers were produced (sha256 recorded below; a rebuild would invalidate
             every answer already on disk)
  harness    the one known way answerers differ - how the request reached them - is
             printed per answerer so it can be read, not assumed

Tests: minesweeper, textworld, roguelike (bench/answers/), certainty (bench/answers/),
and the TextWorld history experiment arms A and B (bench/experiments/tw_history/).

    python3 bench/fairness.py
"""
import hashlib, json, sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
from spec import ENVS, POSITIONS, load_positions  # noqa: E402

MODELS = ["Jev", "OpenJev", "CLM", "Lux", "Nox", "Sol", "Eos", "Kai", "Lex", "Laya", "Haiku-4.5"]
BASELINES = ["heuristic", "oracle", "uniform-random"]
HARNESS = {
    "Jev": "HttpAnswerer -> api.typesafe.ai/v1/systemone (jev-1.13.0), 3 runs",
    "OpenJev": "HttpAnswerer -> vendor helper on the RTX 3090 box, loopback only (shim sha 81a22f1b)",
    "CLM": "HttpAnswerer -> clm-serve on the RTX 3090 box, loopback only (contrastive-lm 0.1.0, head b2b4a8c9)",
    "Haiku-4.5": "subagents reading the question files; format rules + position instructions only",
    **{m: "DecisionAnswerer in-process (as_questions)" for m in ["Lux", "Nox", "Sol", "Eos", "Kai", "Lex"]},
    "Laya": "LayaAnswerer in-process (as_questions)",
}
# playthroughs: every model over HTTP (local ones behind serve_local.py on loopback, OpenJev and CLM
# through ssh tunnels to the RTX 3090 box, loopback only); Haiku through the one-command scripts/tw_turn.sh interface
HIST = BENCH / "experiments" / "tw_history"
# the inputs every answer on disk was produced from; a rebuild invalidates all of them
PINNED = {"certainty.probes.json": "b13c3f4d2eba", "certainty.truth.json": "8a1fc8ae0260",
          "minesweeper.json": "10f65537dc64", "minesweeper.questions.json": "a5ffcd9f6584",
          "roguelike.json": "706b02268cd8", "roguelike.questions.json": "cfe29f3a0342",
          "textworld.json": "1e9971464cb8", "textworld.questions.json": "4795e70c8f6c",
          "textworld_history.json": "a7fa8455eaef"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def main():
    problems, notes = [], []
    pos = {e: {p["pid"]: p for p in load_positions(e)} for e in ENVS}
    probes = json.loads((POSITIONS / "certainty.probes.json").read_text())
    hist = {p["pid"]: p for p in json.loads((HIST / "textworld_history.json").read_text())}
    tests = list(ENVS) + ["certainty", "history A", "history B", "playthrough", "ms struct"]
    msq = {p["pid"]: p for p in json.loads((BENCH / "experiments" / "ms_structured" / "minesweeper_structured.questions.json").read_text())}
    play_manifest = json.loads((BENCH / "experiments" / "tw_play" / "games" / "manifest.json").read_text())
    print(f"{'answerer':14s}" + "".join(f"{t:>12s}" for t in tests))
    for name in MODELS + BASELINES:
        cells = []
        for t in tests:
            if t == "certainty":
                if name in BASELINES:
                    cells.append("n/a"); continue
                f = BENCH / "answers" / f"certainty__{name}.json"
                if not f.exists():
                    cells.append("MISSING"); problems.append(f"{name}: no certainty run"); continue
                said = json.loads(f.read_text())["said"]
                ok = [q for q in (p["qid"] for p in probes) if isinstance(said.get(q), (int, float)) and 0 <= said[q] <= 1]
                bad = len(probes) - len(ok)
                cells.append(f"{len(ok)}/{len(probes)}" + (f" !{bad}" if bad else ""))
                if bad: problems.append(f"{name}: certainty {bad} unanswered/out of range")
                continue
            if t == "ms struct":
                if name in ("oracle", "uniform-random"):
                    cells.append("n/a"); continue
                f = BENCH / "experiments" / "ms_structured" / f"answers__{name}.json"
                if not f.exists():
                    cells.append("MISSING"); problems.append(f"{name}: no structured Minesweeper run"); continue
                pk = json.loads(f.read_text())["picks"]
                miss = sum(1 for k in msq if k not in pk)
                inval = sum(1 for k in msq if k in pk and pk[k] not in {c["id"] for c in msq[k]["candidates"]})
                cells.append(f"{len(msq)-miss-inval}/{len(msq)}" + (f" !{miss+inval}" if miss + inval else ""))
                if miss: problems.append(f"{name}: ms struct {miss} missing")
                if inval: notes.append(f"{name}: ms struct {inval} invalid")
                continue
            if t == "playthrough":
                if name == "oracle":
                    cells.append("n/a"); continue
                f = BENCH / "experiments" / "tw_play" / "results" / f"{name}.json"
                if not f.exists():
                    cells.append("MISSING"); problems.append(f"{name}: no playthrough run"); continue
                gs = {g["seed"]: g for g in json.loads(f.read_text())["games"]}
                done = [e["seed"] for e in play_manifest if e["seed"] in gs and
                        (gs[e["seed"]]["won"] or gs[e["seed"]]["lost"] or gs[e["seed"]]["moves"] >= gs[e["seed"]]["budget"])]
                bad = len(play_manifest) - len(done)
                cells.append(f"{len(done)}/{len(play_manifest)}" + (f" !{bad}" if bad else ""))
                if bad: problems.append(f"{name}: playthrough {bad} games missing or unfinished")
                continue
            if t.startswith("history"):
                if name in ("oracle", "uniform-random"):
                    cells.append("n/a"); continue      # read nothing; history cannot move them
                f = HIST / f"answers__{name}__{t[-1]}.json"
                P = hist
            else:
                f = BENCH / "answers" / f"{t}__{name}.json"
                P = pos[t]
            if not f.exists():
                cells.append("MISSING"); problems.append(f"{name}: no {t} run"); continue
            pk = json.loads(f.read_text())["picks"]
            inval = sum(1 for k in P if k in pk and pk[k] not in {c["id"] for c in P[k]["candidates"]})
            miss = sum(1 for k in P if k not in pk)
            cells.append(f"{len(P) - miss - inval}/{len(P)}" + (f" !{miss + inval}" if miss + inval else ""))
            if miss: problems.append(f"{name}: {t} {miss} missing")
            if inval: notes.append(f"{name}: {t} {inval} invalid (the answerer's own output; graded wrong, shown in the tables)")
        print(f"{name:14s}" + "".join(f"{c:>12s}" for c in cells))
    files = {f.name: f for f in POSITIONS.glob("*.json")} | {"textworld_history.json": HIST / "textworld_history.json"}
    for n, h in PINNED.items():
        if sha(files[n]) != h:
            problems.append(f"input {n} changed since the answers were produced ({sha(files[n])} != {h})")
    print(f"\ninputs: {len(PINNED)} files checked against pinned sha256")
    print("\nharness per answerer:")
    for m in MODELS:
        print(f"  {m:12s} {HARNESS[m]}")
    for n in notes:
        print("  note:", n)
    if problems:
        print("\nNOT FAIR YET:"); [print("  -", p) for p in problems]
        raise SystemExit(1)
    print("\nevery answerer has a complete answer set on every test (invalid answers, if any, noted above and graded wrong)")


if __name__ == "__main__":
    main()
