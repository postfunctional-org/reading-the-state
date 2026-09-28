"""Jev's run-to-run spread: the headline run (bench/answers/) plus runs 2 and 3
here, identical settings. Writes summary.json for the report."""
import itertools, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from spec import ENVS, POSITIONS, load_positions  # noqa: E402


def auroc(pos, neg):
    """P(score on a positive > score on a negative), ties count half"""
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    out = {"runs": 3, "envs": {}, "certainty": []}
    for env in ENVS:
        pos = load_positions(env)
        runs = [json.loads((BENCH / "answers" / f"{env}__Jev.json").read_text())["picks"]]
        runs += [json.loads((HERE / f"{env}__Jev.r{r}.json").read_text())["picks"] for r in (2, 3)]
        rates = [sum(p["pid"] in r and r[p["pid"]] in p["truth"]["optimal_ids"] for p in pos) / len(pos) for r in runs]
        agree = [sum(a.get(p["pid"]) == b.get(p["pid"]) for p in pos) / len(pos)
                 for a, b in itertools.combinations(runs, 2)]
        all3 = sum(len({r.get(p["pid"]) for r in runs}) == 1 for p in pos) / len(pos)
        out["envs"][env] = {"rates": rates, "pairwise_agree": agree, "all_three_agree": all3}
    truth = json.loads((POSITIONS / "certainty.truth.json").read_text())
    runs = [json.loads((BENCH / "answers" / "certainty__Jev.json").read_text())["said"]]
    runs += [json.loads((HERE / f"certainty__Jev.r{r}.json").read_text())["said"] for r in (2, 3)]
    for said in runs:
        safe = [v for q, v in said.items() if v is not None and not truth[q]]
        mine = [v for q, v in said.items() if v is not None and truth[q]]
        out["certainty"].append({"gap": sum(safe) / len(safe) - sum(mine) / len(mine), "auroc": auroc(safe, mine)})
    (HERE / "summary.json").write_text(json.dumps(out, indent=1))
    for env, v in out["envs"].items():
        print(f"{env:12s} rates {' '.join(f'{x:.1%}' for x in v['rates'])}   pairwise agree "
              f"{' '.join(f'{x:.1%}' for x in v['pairwise_agree'])}   all three {v['all_three_agree']:.1%}")
    print("certainty   ", "  ".join(f"gap {c['gap']:+.3f} AUROC {c['auroc']:.3f}" for c in out["certainty"]))


if __name__ == "__main__":
    main()
