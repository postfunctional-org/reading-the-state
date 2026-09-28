"""Diagnostic: is Jev's Minesweeper failure a negation failure?
Original ASCII positions, instruction reworded without negation - identical meaning:
  was: "...Choose the hidden square least likely to be a mine; if one can be proved safe from the numbers, choose that."
  now: "...Choose the hidden square most likely to be safe; if one can be proved safe from the numbers, choose that."
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import answerers
from spec import load_positions
OLD = "Choose the hidden square least likely to be a mine;"
NEW = "Choose the hidden square most likely to be safe;"
a = answerers.build(sys.argv[1])
picks = {}
for p in load_positions("minesweeper"):
    assert OLD in p["instructions"]
    q = dict(p, instructions=p["instructions"].replace(OLD, NEW))
    picks[p["pid"]] = a.pick(q)[0]
(HERE / f"diag__{a.name}.json").write_text(json.dumps({"instructions": NEW, "picks": picks}, indent=1))
print(a.name, "done")
