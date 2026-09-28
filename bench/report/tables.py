"""Print the README's tables as markdown, from bench/report/data.json.

Every table in README.md between a `<!-- table:NAME -->` marker and the matching
`<!-- /table -->` comes from here, so no number in them is typed by hand.

    python3 bench/report/tables.py            # rewrite the tables in README.md in place
    python3 bench/report/tables.py --check    # exit 1 if README.md is out of date
    python3 bench/report/tables.py --print    # print every table
"""
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE.parents[1] / "README.md"
D = json.loads((HERE / "data.json").read_text())
TASKS = ["textworld", "roguelike", "minesweeper"]
TITLE = {"textworld": "TextWorld", "roguelike": "Roguelike", "minesweeper": "Minesweeper"}
SHOWN_BASELINES = {"heuristic", "uniform-random"}


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def pval(p):
    if p is None:
        return "—"
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" if i == 0 else "--:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def name(n):
    return {"heuristic": "hand-written heuristic", "uniform-random": "uniform random"}.get(n, n)


def task(e):
    env = D["envs"][e]; t = env["trivial"]
    rows = []
    for r in env["rows"]:
        if r["kind"] == "baseline" and r["name"] not in SHOWN_BASELINES:
            continue
        lo, hi = r["ci"]
        rows.append([name(r["name"]), pct(r["rate"]), f"{100*lo:.1f}–{100*hi:.1f}",
                     f"{100 * r['delta']:+.1f}", pval(r["p"]), pct(r["easy"]), pct(r["hard"]), pval(r["hard_p"])])
    head = f"{env['n']} positions. Chance {pct(env['chance'])}. State-blind policy: `{t['policy']}`, {pct(t['rate'])}. " \
           f"Hard half: {env['n_hard']} positions, chance {pct(env['chance_hard'])}.\n\n"
    return head + table(["answerer", "optimal", "95% CI", "vs state-blind (pts)", "p", "easy half", "hard half", "hard p vs chance"], rows)


def strata(e):
    env = D["envs"][e]; order = env["strata_order"]
    rows = [[name(r["name"])] + [pct(r["by_stratum"][s]["rate"]) for s in order]
            for r in env["rows"] if r["kind"] != "baseline" or r["name"] in SHOWN_BASELINES]
    return table(["answerer"] + [f"`{s}`" for s in order], rows)


def certainty():
    rows = [[name(r["answerer"]), f"{r['gap']:+.3f}", f"{r['auroc']:.3f}", pval(r["p"]), r["distinct"]]
            for r in sorted(D["certainty"], key=lambda r: -r["auroc"])]
    return table(["answerer", "gap (safe − mine)", "AUROC", "p", "distinct values"], rows)


def calibration():
    rows = []
    for r in D["calibration"]["rows"]:
        cells = []
        for e in TASKS:
            c = r.get(e)
            cells.append("—" if not c else f"{c['auroc']:.2f} / {c['over']:+.2f}")
        rows.append([r["name"]] + cells)
    return table(["model"] + [f"{TITLE[e]} AUROC / over" for e in TASKS], rows)


def crossdevice():
    x = D["crossdevice"]["envs"]; models = [r["model"] for r in x["textworld"]]
    rows = []
    for m in models:
        cells = []
        for e in TASKS:
            r = next(r for r in x[e] if r["model"] == m)
            cells.append(f"{pct(r['agree'])} ({100 * r['delta']:+.1f})")
        rows.append([m] + cells)
    return table(["model"] + [f"{TITLE[e]} agree (Δ pts)" for e in TASKS], rows)


def history():
    rows = []
    for r in D["tw_history"]["rows"]:
        a, b = r["all"]["A"], r["all"]["B"]
        rows.append([name(r["name"]), pct(a["rate"]), pct(b["rate"]), r["all"]["fixed"], r["all"]["broken"], pval(r["all"]["p"]),
                     pct(r["late"]["A"]["rate"]), pct(r["late"]["B"]["rate"])])
    return table(["answerer", "no history", "with history", "fixed", "broken", "p", "late, no history", "late, with history"], rows)


def games():
    rows = []
    for r in sorted(D["tw_play"]["rows"], key=lambda r: -r["won"]):
        lo, hi = r["ci"]
        mo = "—" if r["moves_over_opt"] is None else f"{r['moves_over_opt']:.2f}×"
        rows.append([name(r["name"]), f"{r['won']}/{r['n']}", f"{100*lo:.1f}–{100*hi:.1f}", mo, r["loops"], pct(r["step_acc_with_history"])])
    return table(["player", "won", "95% CI", "moves / optimal (wins)", "games ending in a loop", "single-step accuracy with history"], rows)


def ms_structured():
    s = D["ms_structured"]; rows = []
    for r in s["rows"]:
        a, h = r["all"], r["hard"]
        rows.append([name(r["name"]), pct(a["A"]), pct(a["S"]), pval(a["p"]), pct(h["A"]), pct(h["S"]),
                     pct(r.get("mine_share")) if r.get("mine_share") is not None else "—"])
    head = f"{s['n']} positions. State-blind `{s['trivial']}`: {pct(s['trivial_rate'])}. Hard half chance {pct(s['chance_hard'])}.\n\n"
    return head + table(["answerer", "grid", "structured", "p", "hard, grid", "hard, structured", "provable picks that are mines"], rows)


def ms_wording():
    s = D["ms_wording"]
    rows = [[r["name"], f"{r['p_mine_least']:.2f}", f"{r['p_mine_safe']:.2f}", f"{r['p_mine_mine']:.2f}", pct(r["same_pick_when_goal_reversed"], 0)]
            for r in s["rows"]]
    head = f"Mean true mine probability of the picked square. A random candidate: {s['uniform_p_mine']:.2f}.\n\n"
    return head + table(["model", "“least likely to be a mine”", "“safe”", "“most likely to be a mine”", "same pick when the goal is reversed"], rows)


def repeats():
    j = D["jev_repeats"]["envs"]
    rows = [[TITLE[e], " / ".join(pct(x) for x in j[e]["rates"]), pct(j[e]["all_three_agree"])] for e in TASKS]
    return table(["task", "Jev, three runs", "all three picked the same"], rows)


TABLES = {"textworld": lambda: task("textworld"), "roguelike": lambda: task("roguelike"), "minesweeper": lambda: task("minesweeper"),
          "roguelike_strata": lambda: strata("roguelike"), "certainty": certainty, "calibration": calibration,
          "crossdevice": crossdevice, "history": history, "games": games, "ms_structured": ms_structured,
          "ms_wording": ms_wording, "jev_repeats": repeats}


def render(text):
    def sub(m):
        return f"<!-- table:{m.group(1)} -->\n{TABLES[m.group(1)]()}\n<!-- /table -->"
    return re.sub(r"<!-- table:(\w+) -->.*?<!-- /table -->", sub, text, flags=re.S)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    if a.print:
        for k, f in TABLES.items():
            print(f"### {k}\n\n{f()}\n")
        return
    old = README.read_text(); new = render(old)
    if a.check:
        print("README tables up to date" if new == old else "README tables are stale: run python3 bench/report/tables.py")
        sys.exit(0 if new == old else 1)
    README.write_text(new); print("README.md tables rewritten")


if __name__ == "__main__":
    main()
