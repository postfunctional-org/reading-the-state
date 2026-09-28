# Protocol

How positions are built, how answerers run, and how they are graded. Results are
in `README.md`. A new answerer needs no change to the environments, the ground
truth or the grader.

## Positions

Each task is a set of frozen positions: a state, candidate actions, and an exact
cost per candidate. Answerers see the state and candidates, never the costs.

```
bench/positions/<env>.json            positions with ground truth (grading)
bench/positions/<env>.questions.json  the same positions without it (answering)
bench/answers/<env>__<answerer>.json  one answerer's picks
```

```json
{
  "env": "minesweeper" | "textworld" | "roguelike",
  "pid": "ms0007",
  "meta":  { "seed": 12, "stratum": "guess", ... },
  "state": "the exact text the answerer is shown",
  "instructions": "the Choice question's instructions",
  "candidates": [ { "id": "c4r7", "desc": "col 4 row 7, touching 1,2, 3 hidden",
                    "kind": "interior" } ],
  "truth": {
    "cost":        { "c4r7": 0.0, "c1r2": 0.33 },
    "optimal_ids": [ "c4r7" ],
    "scale":       "probability" | "moves" | "blunder",
    "extra":       { ... }
  }
}
```

`cost` is never negative, lower is better, and 0 is optimal. It always comes from
a solver or an oracle.

| task | cost | ground truth |
|---|---|---|
| minesweeper | excess P(mine) | enumeration of every mine layout consistent with the board |
| textworld | extra moves | TextWorld's `policy_commands` oracle, per candidate via `env.copy()` |
| roguelike | 0 or 1 | hand-built positions where every move but one is a blunder |

## Baselines

- Chance: expected accuracy of a uniform pick, `mean(|optimal_ids| / |candidates|)`.
- Best trivial: the best "always prefer kind K" policy, from
  `spec.trivial_policies()`. It reads nothing about the state.
- Heuristic: a few if-statements per task. No solver, no search.

Beating chance means little. Beating the best trivial policy is the minimum
evidence of reading the state.

Rules enforced in `bench/spec.py`:

- Sets are stratified in equal parts (roguelike 45 per rule, minesweeper 135
  deducible and 135 guess, textworld by distance to goal).
- Trivial policies are computed and printed for every set.
- Candidate order is shuffled per position. A fixed order once added up to 64
  points.

Each set is also split by the best trivial policy. Easy is what it gets right,
hard is what it gets wrong. Quote the hard half. `bench/breakdown.py` prints both
halves, per-stratum accuracy, and a Fisher test of the hard half against chance.

```bash
python3 bench/breakdown.py
python3 bench/breakdown.py --env textworld
```

### Completeness

`bench/grade.py` prints missing and invalid counts and flags uneven gaps:

```
! Haiku-4.5: 18 missing, 0 invalid -> graded per stratum heal_or_die: 27/45
                                      SET IS UNBALANCED FOR THIS ANSWERER
```

This caught a real case: two failed Haiku batches were both `heal_or_die`, which
raised its roguelike score from 76.3% to 82.9%. Fill the gaps before reading the
table.

## Local models

```bash
.venv-enc/bin/python bench/run_answerer.py --answerer Kai --env all   # encoders
.venv/bin/python     bench/run_answerer.py --answerer Nox --env all   # decoders
.venv/bin/python     bench/run_answerer.py --answerer Laya --env all
python3 bench/run_answerer.py --answerer oracle         --env all
python3 bench/run_answerer.py --answerer uniform-random --env all
python3 bench/run_answerer.py --answerer heuristic      --env all
python3 bench/grade.py
```

Answers are saved per task and answerer, so runs are incremental.

The Decision 1.0 bundles go in `models/` (for example `models/Kai-0.6B/`). They
ship for AMD ROCm only. On NVIDIA the block is only a guard, and
`harness/backends.py` patches it in memory without editing `models/`. The
encoders pin `transformers==4.57.6` and the decoders need
`flash-linear-attention`. Each needs its own environment:

```bash
uv venv .venv     && uv pip install --python .venv/bin/python     torch transformers fla-core matplotlib
uv venv .venv-enc && uv pip install --python .venv-enc/bin/python torch "transformers==4.57.6"
uv venv .venv-tw  && uv pip install --python .venv-tw/bin/python  textworld==1.7.0
```

Each run first replays the vendor's model-card example and compares
probabilities (`validate_reference()`).

## Hosted models

`HttpAnswerer` in `bench/answerers.py` speaks the SystemOne request shape. Set it
up from the environment. Never put a key on the command line or in the repo.

```bash
export JEV_BASE_URL="https://api.typesafe.ai"
export JEV_API_KEY="..."            # never commit this
export JEV_MODEL="jev-1.13.0"
export JEV_PATH="/v1/systemone"     # optional, the default

python3 bench/run_answerer.py --answerer http:JEV --env all --limit 5   # smoke test
python3 bench/run_answerer.py --answerer http:JEV --env all
python3 bench/grade.py
```

Request per position:

```http
POST {JEV_BASE_URL}{JEV_PATH}
Authorization: Bearer {JEV_API_KEY}
Content-Type: application/json

{
  "model": "{JEV_MODEL}",
  "state": "<position.state>",
  "questions": {
    "act": {
      "type": "choice",
      "instructions": "<position.instructions>",
      "criteria": { "<candidate id>": "<candidate desc>", ... }
    }
  }
}
```

The pick is `answers.act.choice`. The certainty probe sends a Noul question ("The
hidden square at column C, row R does NOT contain a mine.") and reads
`answers.safe.noul`:

```bash
python3 bench/run_certainty.py --answerer http:JEV     # never pass --rebuild
python3 bench/grade_certainty.py
```

- A pick that is not a candidate counts as invalid. Check `invalid` is 0 on the
  smoke test.
- A full pass is 629 positions, 400–1,100 input tokens each.
- `http:NAME` reads `NAME_BASE_URL`, `NAME_API_KEY` and `NAME_MODEL`.
- For a different API shape, override `path` and `HttpAnswerer.read_choice`.
- An API key may be omitted only for a loopback URL.

### OpenJev

`scripts/serve_openjev.sh` is the exact setup behind every OpenJev number: the
FP8 build under vLLM with 12 GiB offloaded, plus the vendor's helper on port 3111.

```bash
OPENJEV_BASE_URL=http://127.0.0.1:3111 OPENJEV_MODEL=openjev \
  python bench/run_answerer.py --answerer http:OpenJev --env all
```

- Keep the helper on `127.0.0.1`. It has no authentication without `SHIM_TOKEN`.
- `--max-num-seqs 8`. Each sequence holds about 75 MiB of state, and the vendor's
  256 would need 18.7 GiB.
- `VLLM_USE_FLASHINFER_SAMPLER=0` avoids a CUDA toolkit build the helper never
  uses.
- `READOUT_TARGETED=1` is required. The calibration was fitted with it on.
- `GET /v1/version` returns the helper's sha256 and calibration. It is stored with
  the answers.
- There is no reference output to check against. After the engine crashed, all
  207 earlier answers were asked again and matched. Do this after any crash.

### CLM

`scripts/serve_clm_encoder.sh` starts the Qwen3-8B embedding server and
`scripts/serve_clm.sh` starts `clm-serve`. `clm-serve` binds to 0.0.0.0 with no
authentication by default. Always pass `--host 127.0.0.1`.

## Rebuilding positions

```bash
python3             bench/export_minesweeper.py   # 135 + 135
python3             bench/export_roguelike.py     # 45 x 3
.venv-tw/bin/python bench/export_textworld.py     # TextWorld 1.7.0
python3 bench/run_certainty.py --answerer X --rebuild   # new certainty probes
```

- Rebuilding a set invalidates its answers. Delete `bench/answers/<env>__*.json`
  and rerun every answerer.
- The certainty probes are frozen separately, and `run_certainty.py` refuses to
  score orphaned probes.
- Everything is seeded. Rebuilds are byte-identical.
- Size a set by its hard half. Minesweeper at 45 + 45 left 60 hard positions,
  too few. At 135 + 135 it has 194.

## Second machine

```bash
rsync -a --exclude='.venv*' --exclude='models' ./ user@host:~/reading-the-state/
rsync -a models/ user@host:~/reading-the-state/models/
rsync -a --dry-run --itemize-changes models/ user@host:~/reading-the-state/models/   # must print nothing
# run the answerers there, then:
rsync -a user@host:~/reading-the-state/bench/answers/ bench/answers-other/
python3 bench/crossdevice.py --other bench/answers-other --label "RTX 3090"
```

Published comparison, RTX 4070 Ti SUPER against RTX 3090, in
`bench/answers-3090/`:

| | agreement | accuracy change |
|---|---|---|
| encoders (Kai, Lex) | 100% | 0.0 |
| decoders and Laya | 94.8%–99.6% | within ±1.5 points |
| baselines | 100% | 0.0 |

- Only compare answerers that were actually rerun. A copied file shows 100%
  agreement. Haiku cannot be rerun locally.
- Decoder disagreement follows the model's margin between its top two choices
  (r = +0.59 over 15 task-model pairs). Encoders match at any margin.

### Traps

- Triton needs Python headers. Without `Python.h`, FLA silently falls back to the
  CPU. Use `uv python install 3.12`, and fail on the warning `roll back to CPU`.
- Seed baselines per position. One RNG per run made two identical machines
  disagree on 94% of random picks. `bench/answerers.py::_rng_for` does this.

## Known limits

- Roguelike: no general solver. Only forced positions are graded. It measures
  blunder avoidance.
- Certainty probe: Minesweeper only, 120 safe and 120 mined squares. Report both
  the gap (is the number usable) and the AUROC (is the order right).
- Token limits: the encoders take 1,024 tokens, the decoders 16,384. Sets are
  built to fit 1,024.
- Minesweeper mixes two skills: finding a square in the grid, and reasoning about
  it. Haiku 4.5 misreads the grid. Swapping its column and row lifts it from
  18.4% to 37.9% on 87 positions (Fisher p = 0.0067). Jev and OpenJev do not.
- Lux-9B: its 14.78 GiB backbone does not fit, so part of it streams from host
  RAM (`harness/backends.py::_backbone_device_map`, `LUX_GPU_GIB` to override).
  Dtypes are unchanged, and two different splits gave bit-identical outputs. It
  runs at about 1.4 positions per second against Nox's 12.
  `DecoderBackend._placement` checks that every module executes on the GPU.
- Reference agreement: worst probability difference against the vendor's example
  is Nox 6.8e-04, Eos 1.3e-03, Sol 4.0e-03, Lux 2.1e-02, with no argmax flips.
  Lux's gap comes from the platform. The offload plays no part.
- Haiku can reason before answering and the decision models cannot. That favours
  Haiku.
