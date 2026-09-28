"""Minesweeper with the board pre-processed into structured text, the way the
Mario and Doom harnesses feed Jev: code extracts the objects and their
relationships; the model makes the decision.

The line held here: the harness DESCRIBES the board, it never SOLVES it.
Each revealed number is listed with the hidden squares that touch it - which
removes reading coordinates off an ASCII grid - and nothing else: no probabilities,
no squares marked safe or mined, no subtraction of known mines. Header line,
instructions, candidates and ground truth are the frozen position's, byte for byte.

    python3 bench/experiments/ms_structured/build.py
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from spec import load_positions  # noqa: E402


def parse(state):
    lines = state.split("\n")
    header = lines[0]
    rows = [l for l in lines[1:] if l.strip() and l.strip()[0].isdigit() and len(l.split()) > 2 and not l.startswith("    ")]
    grid = {}
    for l in rows:
        parts = l.split()
        r = int(parts[0])
        for c, ch in enumerate(parts[1:]):
            grid[(c, r)] = ch
    return header, grid


def structured(state):
    """Header, one explanatory line, then one line per distinct constraint the
    revealed numbers impose: "N: <hidden squares>" = exactly N mines among them. A number N
    touching hidden set H means exactly that (there are no flags), so this restates
    the board without inferring anything; identical constraints from several
    numbers appear once."""
    header, grid = parse(state)
    cons = {}
    for (c, r), ch in sorted(grid.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if not ch.isdigit():
            continue
        hid = tuple(f"c{c+dc}r{r+dr}" for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dc or dr) and grid.get((c + dc, r + dr)) == ".")
        if hid:
            cons.setdefault((int(ch), hid), None)
    out = [header, "",
           "What the revealed numbers say, one line per constraint. \"N: squares\" means exactly N mines among those hidden squares. cXrY is the square at col X row Y."]
    for n, hid in cons:
        out.append(f"{n}: {' '.join(hid)}")
    return "\n".join(out)


def main():
    pos = load_positions("minesweeper")
    out = []
    for p in pos:
        _, grid = parse(p["state"])
        hidden = {f"c{c}r{r}" for (c, r), ch in grid.items() if ch == "."}
        assert {c["id"] for c in p["candidates"]} <= hidden, p["pid"]      # parse agrees with the candidates
        q = dict(p); q["state"] = structured(p["state"]); out.append(q)
    (HERE / "minesweeper_structured.json").write_text(json.dumps(out, indent=1))
    print(len(out), "positions"); print(out[5]["state"])


if __name__ == "__main__":
    main()
