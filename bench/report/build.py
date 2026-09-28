"""Collect the graders' output into one file, bench/report/data.json.

Every number in README.md comes from here, never from a hand-edited file:

    bench/report.json, bench/breakdown.json    grade.py, breakdown.py
    bench/certainty.json                        grade_certainty.py
    bench/crossdevice.json                      crossdevice.py
    bench/experiments/*/summary.json            each side experiment's grader

    python3 bench/report/build.py                         # write data.json
    python3 bench/report/build.py --exclude Jev --check   # diff against the committed data.json
"""
import argparse, json, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent

MODELS = {"Kai": "Decision 1.0 · encoder · 0.6B", "Lex": "Decision 1.0 · encoder · 0.6B",
          "Eos": "Decision 1.0 · decoder · 0.8B", "Sol": "Decision 1.0 · decoder · 2B",
          "Nox": "Decision 1.0 · decoder · 4B", "Lux": "Decision 1.0 · decoder · 9B",
          "Laya": "Convai Laya · 0.42B", "OpenJev": "OpenJev · 25B · open weights",
          "Haiku-4.5": "Claude Haiku 4.5 · generative LLM", "Jev": "TypeSafe Jev · hosted",
          "CLM": "Contrastive-LM CLM-v0.1 · 8B · open weights"}


def margin(m, e):
    p = BENCH / "answers" / f"{e}__{m}.json"
    if not p.exists():
        return None
    v = [pr[0] - pr[1] for t in json.loads(p.read_text()).get("telemetry", {}).values()
         if len(pr := sorted((t.get("probs") or {}).values(), reverse=True)) >= 2]
    return statistics.mean(v) if v else None


def build(exclude=()):
    rep = json.loads((BENCH / "report.json").read_text())
    brk = json.loads((BENCH / "breakdown.json").read_text())
    cert = json.loads((BENCH / "certainty.json").read_text())
    xdev = json.loads((BENCH / "crossdevice.json").read_text())
    envs = {}
    for e, ev in rep["envs"].items():
        meta = brk[e]["__meta__"]
        rows = []
        for name, per in rep["answerers"].items():
            a = per.get(e)
            if a is None or name in exclude:
                continue
            b = brk[e].get(name, {})
            rows.append({"name": name, "kind": a["kind"], "rate": a["rate"], "ci": a["ci"],
                         "cost": a["mean_cost"], "delta": a["vs_trivial"]["delta"],
                         "p": a["vs_trivial"]["p"], "missing": a["missing"], "invalid": a["invalid"],
                         "easy": b.get("easy"), "hard": b.get("hard"), "hard_p": b.get("hard_p_vs_chance"),
                         "hard_ci": b.get("hard_ci"), "by_stratum": a["by_stratum"]})
        rows.sort(key=lambda r: -r["rate"])
        envs[e] = {"n": ev["n_positions"], "strata": ev["strata"], "scale": ev["scale"],
                   "chance": ev["chance_rate"], "chance_cost": ev["chance_cost"],
                   "mean_candidates": ev["mean_candidates"], "trivial": meta["best_trivial"],
                   "trivial_all": ev["trivial_policies"], "n_easy": meta["n_easy"],
                   "n_hard": meta["n_hard"], "chance_hard": meta["chance_hard"],
                   "strata_order": sorted(ev["strata"]), "rows": rows}
    data = {"envs": envs,
            "certainty": [c for c in cert if c["answerer"] not in exclude],
            "models": {k: v for k, v in MODELS.items() if k not in exclude},
            "crossdevice": xdev,
            "margins": {e: {m: v for m in MODELS if m not in exclude and m != "Haiku-4.5"
                            and (v := margin(m, e)) is not None} for e in envs}}
    for f in sorted((BENCH / "experiments").glob("*/summary.json")):
        data[f.parent.name] = json.loads(f.read_text())
    return data


def diff(a, b, path=""):
    """yield the paths where two JSON values differ, floats to 1e-12"""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                yield f"{path}/{k} only in {'new' if k in a else 'old'}"
            else:
                yield from diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, (int, float)):
        if abs(a - b) > 1e-12:
            yield f"{path}: {b} -> {a}"
    elif a != b:
        yield f"{path}: {str(b)[:60]} -> {str(a)[:60]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--check", action="store_true", help="diff against data.json, write nothing")
    args = ap.parse_args()
    data = build(set(args.exclude))
    if args.check:
        old = json.loads((HERE / "data.json").read_text())
        d = list(diff(data, old))
        print(f"{len(d)} differences from the committed data.json")
        for x in d[:40]:
            print("  ", x)
        raise SystemExit(0)
    blob = json.dumps(data)
    (HERE / "data.json").write_text(blob)
    print(f"wrote data.json ({len(blob)/1024:.0f} KB)")
