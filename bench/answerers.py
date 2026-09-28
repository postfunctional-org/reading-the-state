"""Answerers: anything that can look at a position and pick a candidate.

Adding a new one - including a hosted model such as Jev - means implementing a
single method:

    class MyAnswerer:
        name = "Jev"
        kind = "model"                       # "model" | "baseline"
        def pick(self, position: dict) -> tuple[str, dict]:
            '''return (candidate id, telemetry dict)'''

and registering it in `build()` below. Everything else - the positions, the
grading, the baselines, the report - is shared, so a new answerer is directly
comparable to every existing one with no other change.

For a hosted SystemOne-compatible endpoint, `HttpAnswerer` already does this; see
PROTOCOL.md for the exact request it sends and how to point it at a provider.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))


def as_questions(position: dict) -> dict:
    """The Choice question for a position, in SystemOne form."""
    return {
        "act": {
            "type": "choice",
            "instructions": position["instructions"],
            "criteria": {c["id"]: c["desc"] for c in position["candidates"]},
        }
    }


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class DecisionAnswerer:
    """Decision 1.0 (Kai, Lex, Eos, Sol, Nox) running locally."""

    kind = "model"

    def __init__(self, name: str):
        from backends import open_backend

        self.name = name
        self.be = open_backend(name)

    def pick(self, position):
        t0 = time.time()
        res = self.be.evaluate({"model": self.be.model_id,
                                "state": position["state"],
                                "questions": as_questions(position)})
        a = res["answers"]["act"]
        return a["choice"], {"confidence": a.get("confidence"),
                             "probs": {k: round(v, 6) for k, v in a["probabilities"].items()},
                             "latency_ms": round(1000 * (time.time() - t0), 1)}


class LayaAnswerer:
    """convaiinnovations/laya - a different vendor, same category and interface."""

    kind = "model"
    name = "Laya"

    def __init__(self, max_len: int = 4096):
        from laya import Router

        self.router = Router()
        self.max_len = max_len

    def pick(self, position):
        t0 = time.time()
        try:
            res = self.router.predict(position["state"], as_questions(position),
                                      max_len=self.max_len)
        except TypeError:
            res = self.router.predict(position["state"], as_questions(position))
        a = res["answers"]["act"]
        return a["choice"], {"confidence": a.get("confidence"),
                             "probs": {k: round(float(v), 6)
                                       for k, v in (a.get("probabilities") or {}).items()},
                             "routed": (res.get("routing") or {}).get("model"),
                             "latency_ms": round(1000 * (time.time() - t0), 1)}


class HttpAnswerer:
    """Any SystemOne-compatible HTTP endpoint. This is the path for Jev.

    Sends exactly the request the Decision 1.0 model cards document:

        POST {base_url}{path}
        Authorization: Bearer {api_key}
        Content-Type: application/json
        {"model": ..., "state": ..., "questions": {"act": {"type": "choice", ...}}}

    and reads `answers.act.choice` from the response. If a provider differs, the
    only things that should need changing are `path` and `read_choice`.
    """

    kind = "model"

    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 path: str = "/v1/systemone", timeout: float = 180.0,
                 retries: int = 1):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.path = path
        self.timeout = timeout
        self.retries = retries

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:                      # a loopback helper has no token
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def read_choice(payload: dict) -> tuple[str, dict]:
        a = payload["answers"]["act"]
        return a["choice"], {"confidence": a.get("confidence"),
                             "probs": a.get("probabilities")}

    def pick(self, position):
        import urllib.error
        import urllib.request

        body = json.dumps({"model": self.model, "state": position["state"],
                           "questions": as_questions(position)}).encode()
        t0 = time.time()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base_url + self.path, data=body,
                                         method="POST", headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode())
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                # One retry: a dropped connection would otherwise land in the
                # grader as `missing`, which silently unbalances the set.
                if attempt == self.retries:
                    raise
                time.sleep(2.0)
        choice, info = self.read_choice(payload)
        info["latency_ms"] = round(1000 * (time.time() - t0), 1)
        info["usage"] = payload.get("usage")
        info["served_model"] = payload.get("model")   # receipt: what actually answered
        return choice, info


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def _rng_for(position, seed: int = 0):
    """A generator seeded from the position id, not from run order.

    A single RNG threaded through a run makes a baseline's answers depend on how
    that run was sliced - `--env all` and three separate `--env` calls produce
    different draws for the same position. Seeding per position makes every
    baseline byte-reproducible on any machine, in any slicing, which is what the
    frozen-position contract is supposed to guarantee for every answerer and not
    only for the models.
    """
    import hashlib
    import random

    h = hashlib.sha256(f"{seed}:{position['env']}:{position['pid']}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


class OracleAnswerer:
    """Perfect play. Reads the ground truth, so it is the ceiling, not a competitor."""

    kind = "baseline"
    name = "oracle"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def pick(self, position):
        return _rng_for(position, self.seed).choice(position["truth"]["optimal_ids"]), {}


class UniformAnswerer:
    kind = "baseline"
    name = "uniform-random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def pick(self, position):
        return _rng_for(position, self.seed).choice(position["candidates"])["id"], {}


class HeuristicAnswerer:
    """A cheap reader: uses the state but runs no solver and no search.

    This is the bar that matters in practice. A model that cannot beat a few
    lines of if-statements on a task is not doing anything useful on that task,
    however far above uniform random it lands.
    """

    kind = "baseline"
    name = "heuristic"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def pick(self, position):
        self.rng = _rng_for(position, self.seed)
        env = position["env"]
        cands = position["candidates"]
        if env == "minesweeper":
            return self._mines(position, cands), {}
        if env == "roguelike":
            return self._rogue(position, cands), {}
        return self._textworld(position, cands), {}

    def _mines(self, position, cands):
        # prefer the square touching the lowest revealed numbers per hidden neighbour
        best, key = None, None
        for c in cands:
            d = c["desc"]
            try:
                nums = d.split("touching", 1)[1].split(",")[0].strip()
                vals = [int(x) for x in nums.split(",") if x.strip().isdigit()] or [9]
            except Exception:
                vals = [9]
            try:
                hidden = int(d.rsplit(",", 1)[1].strip().split()[0])
            except Exception:
                hidden = 1
            k = (sum(vals) / max(1, hidden), max(vals), self.rng.random())
            if key is None or k < key:
                best, key = c["id"], k
        return best

    def _rogue(self, position, cands):
        state = position["state"]
        hp = maxhp = None
        for line in state.splitlines():
            if line.startswith("HP "):
                try:
                    a, b = line.split()[1].split("/")
                    hp, maxhp = int(a), int(b)
                except Exception:
                    pass
                break
        have = {c["kind"]: c for c in cands}
        if "pickup" in have:
            return have["pickup"]["id"]
        if hp is not None and maxhp and hp <= 0.35 * maxhp and "quaff" in have:
            return have["quaff"]["id"]
        if "attack" in have:
            return have["attack"]["id"]
        if "descend" in have:
            return have["descend"]["id"]
        return self.rng.choice(cands)["id"]

    def _textworld(self, position, cands):
        # prefer a command whose object is named in the objective, then act over inspect
        obj = position["state"].split("\n", 1)[0].lower()
        scored = []
        for c in cands:
            words = [w for w in c["id"].lower().split()[1:] if len(w) > 3]
            overlap = sum(1 for w in words if w in obj)
            rank = {"take": 0, "manipulate": 1, "put": 2, "go": 3,
                    "consume": 4, "inspect": 5, "other": 4}.get(c["kind"], 4)
            scored.append((-overlap, rank, self.rng.random(), c["id"]))
        scored.sort()
        return scored[0][3]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def build(name: str):
    """Construct an answerer by name. Extend here to add a provider."""
    low = name.lower()
    if low in ("oracle", "uniform-random", "heuristic"):
        return {"oracle": OracleAnswerer, "uniform-random": UniformAnswerer,
                "heuristic": HeuristicAnswerer}[low]()
    if low.startswith("laya"):
        return LayaAnswerer()
    if low.startswith("http:"):
        # http:NAME  - configured entirely from the environment, so no secret
        # ever lands in a command line or in this repository.
        tag = name.split(":", 1)[1] or "JEV"
        up = tag.upper()
        base = os.environ.get(f"{up}_BASE_URL")
        key = os.environ.get(f"{up}_API_KEY")
        model = os.environ.get(f"{up}_MODEL", tag)
        path = os.environ.get(f"{up}_PATH", "/v1/systemone")
        # A key is mandatory for anything off-box. A helper bound to loopback has
        # no token to send - and requiring a fake one would push people towards
        # exposing the port with a token instead of not exposing it at all.
        from urllib.parse import urlparse

        host = (urlparse(base).hostname or "") if base else ""
        loopback = host in ("127.0.0.1", "::1", "localhost")
        if not base or (not key and not loopback):
            raise SystemExit(
                f"set {up}_BASE_URL and {up}_API_KEY (and optionally {up}_MODEL, "
                f"{up}_PATH) to run the {tag} endpoint - see PROTOCOL.md. "
                f"{up}_API_KEY may be omitted only for a loopback endpoint.")
        return HttpAnswerer(tag, base, key, model, path=path)
    return DecisionAnswerer(name)
