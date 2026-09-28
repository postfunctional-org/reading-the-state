"""Backends that run Decision 1.0 locally on NVIDIA.

The vendor ships these models as AMD ROCm-only and states that no NVIDIA
validation is claimed. Reading the runtime shows the coupling is thinner than the
claim: ``decision_runtime/native.py::_amd_device`` refuses any device where
``torch.version.hip is None``, but nothing underneath it is a HIP kernel. The
encoder branch is stock ``transformers`` ModernBERT in fp32 with SDPA attention,
and ``rocm_contiguous_layout`` is a tensor-layout policy, not a device kernel.

So we replace that one function at runtime. We do not touch a single file in the
model repositories - we cannot, because ``verify_files`` hashes every ``.py`` in
``native/`` and compares the roster against ``RUNTIME_SHA256`` before it will load
anything. Everything the vendor checks still gets checked.

The honest caveat, which belongs in any result produced with this: these are not
the vendor's qualified numbers. ``validate_device_agreement`` below is the
evidence offered in their place - it runs the same request on CPU and on CUDA and
reports the largest probability difference. Agreement to fp32 noise means no
device-specific path is changing the answer.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

# Native identifiers are UNSIZED - the Hub's size suffix is a deployment alias only.
NATIVE = {
    "Kai": dict(dir="Kai-0.6B", sha="c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4"),
    "Lex": dict(dir="Lex-0.6B", sha="f288d873999832a3f37c6a7c4268c2ab309691e621794dbf7acab891acbbb7e6"),
}

_RUNTIME_READY = False


def _install_shim() -> None:
    """Import the encoder runtime once and neutralise its ROCm-only device guard."""
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return
    root = MODELS / NATIVE["Kai"]["dir"]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    import decision_runtime.native as dn

    def _any_cuda_device(device):
        selected = torch.device(device)
        if selected.type == "cpu":
            return selected
        if selected.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("expected a cuda or cpu device")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        return selected

    dn._amd_device = _any_cuda_device
    _RUNTIME_READY = True


class EncoderBackend:
    """Kai-0.6B / Lex-0.6B. Three-path bidirectional encoder, fp32, 1024-token budget."""

    branch = "encoder"

    def __init__(self, name: str, device: str = "cuda:0"):
        if name not in NATIVE:
            raise KeyError(f"unknown encoder model {name!r}; have {sorted(NATIVE)}")
        _install_shim()
        from decision_runtime import load_native
        from decision_inference import SystemOne

        self.name = name
        self.model_id = f"Decision-1.0-{name}"
        self.device = device
        spec = NATIVE[name]
        self.native = load_native(
            str(MODELS / spec["dir"] / "native"),
            expected_manifest_sha256=spec["sha"],
            device=device,
        )
        self.client = SystemOne(self.native, model=None, batching="default")

    def evaluate(self, request: dict) -> dict:
        return self.client.evaluate(request)

    def batch(self, requests: list[dict]) -> list[dict]:
        return self.client.batch(requests)


DECODERS = {
    "Eos": dict(dir="Eos-0.8B", layout="flat"),    # decision/ at the repo root
    "Sol": dict(dir="Sol-2B", layout="src"),       # src/decision/ + a runtime profile guard
    "Nox": dict(dir="Nox-4B", layout="src"),
    # Lux's BF16 backbone is 14.78 GiB and the card is 16 GiB with a desktop on
    # it, so this one spills layers to host RAM. See _backbone_device_map.
    "Lux": dict(dir="Lux-9B", layout="src", offload=True),
}

_MAX_MEMORY = None   # set by DecoderBackend before the vendor loader runs


ACTIVATION_HEADROOM_GIB = 3.0   # measured: well above what these models use at
                                # benchmark question sizes, with room for fragmentation


def _backbone_gib(root: Path) -> float | None:
    """Size of the bundle's backbone from its safetensors index, in GiB."""
    index = root / "backbone" / "model.safetensors.index.json"
    if not index.exists():
        return None
    total = json.loads(index.read_text()).get("metadata", {}).get("total_size")
    return (total / 2 ** 30) if total else None


def _offload_budget(root: Path) -> dict | None:
    """An accelerate max_memory map, or None when the model simply fits.

    Offload is a last resort, not a default: it costs roughly 8x throughput. So
    this compares the bundle's actual backbone size against the VRAM actually free
    and only spills when it has to - which means the same code runs resident on a
    24 GiB card and offloaded on a 16 GiB one with a desktop on it, with no flag.

    transformers' caching_allocator_warmup pre-reserves the GPU share of the map
    as one contiguous block, so the GPU budget is set well under free memory rather
    than up to it. DECISION_GPU_GIB forces a budget (and forces offload);
    DECISION_NO_OFFLOAD=1 refuses to offload at all.
    """
    import torch

    free = torch.cuda.mem_get_info()[0] / 2 ** 30
    need = _backbone_gib(root)

    override = os.environ.get("DECISION_GPU_GIB") or os.environ.get("LUX_GPU_GIB")
    if override:
        return {0: f"{int(override)}GiB", "cpu": "40GiB"}
    if need is None or need + ACTIVATION_HEADROOM_GIB <= free:
        return None                      # it fits; run it resident
    if os.environ.get("DECISION_NO_OFFLOAD"):
        raise RuntimeError(
            f"backbone is {need:.2f} GiB and only {free:.2f} GiB is free; "
            f"offload is required but DECISION_NO_OFFLOAD is set")
    gib = max(2, min(math.floor(free - 2.0), math.ceil(need)))
    return {0: f"{gib}GiB", "cpu": "40GiB"}


@contextlib.contextmanager
def _backbone_device_map(max_memory: dict):
    """Give the vendor's backbone load an accelerate device_map, in memory only.

    The bundle has no device_map plumbing at all: ``code/decision_model.py`` calls
    ``Qwen3_5TextModel.from_pretrained(...)`` with no placement argument, so the
    weights land in host RAM, and ``code/decision_api.py`` then does
    ``model.to(device)``. That second line is where a 14.78 GiB backbone dies on a
    16 GiB card - not at load. This patches the two halves of that:

    * ``from_pretrained`` is replaced on the *transformers* class, which is what
      the bundle resolves at call time. Nothing under models/ is touched, and it
      survives decision_api.py re-importing decision_model.py by path on every
      construction.
    * ``model.to(device)`` is plain ``nn.Module.to`` and would recurse into the
      backbone and drag every offloaded layer back onto the card. Neutering
      ``_apply`` on the backbone instance stops the recursion exactly there, so the
      small FP32 head still moves and accelerate's placement survives. ``.eval()``
      is unaffected - it recurses through ``children()``, not ``_apply``.

    Offload is a device change only: offloaded parameters become meta tensors that
    keep their dtype, so the bundle's own "backbone must remain BF16 / head must
    remain FP32" assertions still run and still pass.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    original = Qwen3_5TextModel.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("device_map", "auto")
        kwargs.setdefault("max_memory", max_memory)
        backbone = original(cls, *args, **kwargs)
        backbone._apply = lambda *a, **k: backbone
        return backbone

    Qwen3_5TextModel.from_pretrained = classmethod(from_pretrained)
    try:
        yield
    finally:
        Qwen3_5TextModel.from_pretrained = classmethod(original)


class DecoderBackend:
    """Eos / Sol / Nox / Lux. Qwen3.5 backbone (BF16) + FP32 candidate head, 16384-token budget.

    Only one of these can live in a process: every repo ships a package literally
    named ``decision``. Run one model per process.

    Sol/Nox/Lux check the runtime against the bundle they were qualified on and
    refuse to load when it differs. On NVIDIA it always differs - ``torch.version.hip``
    is None here - so we pass the vendor's own documented escape hatch,
    ``allow_unvalidated_runtime=True``, which downgrades the refusal to a warning.
    RUNTIME.md is explicit that this marks an exploratory run, not a benchmark
    reproduction, and results produced this way are labelled accordingly.
    """

    branch = "decoder"

    def __init__(self, name: str, device: str = "cuda:0", batch_size: int = 8,
                 max_memory: dict | None = None):
        global _MAX_MEMORY
        if name not in DECODERS:
            raise KeyError(f"unknown decoder model {name!r}; have {sorted(DECODERS)}")
        spec = DECODERS[name]
        root = MODELS / spec["dir"]
        # `offload` marks a bundle that may not fit; _offload_budget decides whether
        # it actually has to on this machine, and returns None when it fits.
        if max_memory is None and spec.get("offload"):
            max_memory = _offload_budget(root)
        _MAX_MEMORY = max_memory
        self.max_memory = max_memory
        path = str(root / "src") if spec["layout"] == "src" else str(root)
        if path not in sys.path:
            sys.path.insert(0, path)

        self.name = name
        self.model_id = f"Decision-1.0-{name}"
        self.device = device

        if spec["layout"] == "flat":
            from decision.engine import DecisionModel

            self.model = DecisionModel.from_pretrained(str(root), device=device, batch_size=batch_size)
            self.runtime_note = "no runtime profile guard in this bundle"
        else:
            import warnings

            import decision.model as dmod
            from decision.model import DecisionModel

            # Sol/Nox bind an FLA autotune profile (l2norm_fwd_kernel.json) that is
            # hard-bound to gfx942 and loaded under FLA_CACHE_MODE=strict, which
            # forbids any fallback. On NVIDIA `gcnArchName` does not exist, so the
            # bind raises before the model is even read. Those pinned launch configs
            # (BT / num_warps) are tuned for AMD anyway - forcing them onto CUDA
            # would be worse than not binding them. So we skip the bind and let FLA
            # autotune for this GPU, which is the portable path.
            # Acceptance test for this choice: models/<name>/model-card-example.json
            # ships the vendor's exact reference probabilities. See validate_reference().
            if not getattr(dmod, "_decision_profile_bypassed", False):
                _orig_load_api = dmod._load_api

                def _load_api_no_profile(path):
                    api = _orig_load_api(path)
                    api.prepare_runtime_profile = lambda checkpoint, device="cuda:0": None
                    if _MAX_MEMORY is not None:
                        engine_cls = api.DecisionEngine

                        class _OffloadEngine(engine_cls):
                            def __init__(self, *a, **kw):
                                with _backbone_device_map(_MAX_MEMORY):
                                    super().__init__(*a, **kw)

                        api.DecisionEngine = _OffloadEngine
                    return api

                dmod._load_api = _load_api_no_profile
                dmod._decision_profile_bypassed = True

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.model = DecisionModel.from_pretrained(
                    str(root),
                    device=device,
                    local_files_only=True,
                    allow_unvalidated_runtime=True,
                )
            self.runtime_note = "; ".join(str(w.message)[:400] for w in caught) or "no warning"

        self.placement = self._placement()

    def _placement(self) -> dict:
        """Record where the backbone actually ended up.

        The `_apply` neuter in _backbone_device_map leans on nn.Module.to routing
        through _apply. If that ever stops holding, the backbone is silently
        dragged onto the card; this is the one line that would catch it.
        """
        import torch

        engine = getattr(self.model, "_engine", None)
        backbone = getattr(getattr(engine, "model", None), "backbone", None)
        if backbone is None:
            return {}
        dmap = getattr(backbone, "hf_device_map", None) or {}
        devs = [p.device.type for p in backbone.parameters()]

        # Weights may live on the host; EXECUTION must not. FLA's gated-delta-rule
        # is a Triton kernel with no CPU path, so a layer accelerate decided to run
        # on the CPU fails with "Pointer argument cannot be accessed from Triton"
        # thirty layers deep, in a trace that names nothing about device maps.
        # Checking it here turns that into one legible error at load time.
        bad = []
        for mod_name, mod in backbone.named_modules():
            hook = getattr(mod, "_hf_hook", None)
            exec_dev = getattr(hook, "execution_device", None)
            if exec_dev is not None and torch.device(exec_dev).type != "cuda":
                bad.append(f"{mod_name}->{exec_dev}")
        if bad:
            raise RuntimeError(
                f"{self.name}: {len(bad)} module(s) would EXECUTE off the GPU "
                f"({', '.join(bad[:4])}...). Offloaded weights are fine, offloaded "
                f"execution is not - FLA has no CPU kernel. Lower the GPU budget so "
                f"accelerate streams weights instead (LUX_GPU_GIB).")
        out = {"modules_on_cpu": sum(1 for v in dmap.values() if v in ("cpu", "disk")),
               "modules_on_gpu": sum(1 for v in dmap.values() if v not in ("cpu", "disk")),
               "params_offloaded": sum(1 for d in devs if d == "meta"),
               "params_resident": sum(1 for d in devs if d == "cuda"),
               "vram_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2)}
        if self.max_memory is not None and not (out["modules_on_cpu"] or out["params_offloaded"]):
            raise RuntimeError(
                f"{self.name}: offload was requested ({self.max_memory}) but nothing is "
                f"offloaded - the _apply neuter in _backbone_device_map is no longer holding")
        return out

    def evaluate(self, request: dict) -> dict:
        return self.model.decide(state=request["state"], questions=request["questions"])

    def batch(self, requests: list[dict]) -> list[dict]:
        inner = getattr(self.model, "decide_batch", None)
        if inner is None:
            inner = getattr(self.model, "_engine").decide_batch
        return inner([{"state": r["state"], "questions": r["questions"]} for r in requests])


def validate_reference(name: str) -> dict:
    """Replay the vendor's shipped model-card example and diff against its recorded answer.

    This is the acceptance gate for running these models off their qualified AMD
    runtime: the vendor published exact probabilities, so we can check ours.
    """
    root = MODELS / DECODERS[name]["dir"]
    card = json.loads((root / "model-card-example.json").read_text())

    # Two bundle shapes ship in the wild. Eos/Sol/Nox carry a single
    # request/response pair; Lux carries `requests`/`actual_responses` keyed by
    # example name. Normalise to a list of (label, request, response).
    if "request" in card:
        cases = [("model_card", card["request"], card["response"])]
    else:
        cases = [(k, card["requests"][k], card["actual_responses"][k])
                 for k in card["requests"]]

    be = DecoderBackend(name)
    worst, compared, flips, tokens = 0.0, 0, [], []
    for label, req, ref_resp in cases:
        got = be.evaluate({"state": req["state"], "questions": req["questions"]})
        tokens.append({"case": label,
                       "ref": ref_resp.get("usage", {}).get("input_tokens"),
                       "got": got.get("usage", {}).get("input_tokens")})
        for qid, ref in ref_resp["answers"].items():
            mine = got["answers"][qid]
            if ref["type"] == "noul":
                worst = max(worst, abs(ref["noul"] - mine["noul"]))
                compared += 1
            else:
                for k, v in ref["probabilities"].items():
                    worst = max(worst, abs(v - mine["probabilities"][k]))
                    compared += 1
                if ref["type"] == "choice" and ref["choice"] != mine["choice"]:
                    flips.append(f"{label}.{qid}")
    return {
        "model": name,
        "n_cases": len(cases),
        "max_abs_prob_delta": worst,
        "n_compared": compared,
        "argmax_flips": flips,
        "input_tokens": tokens,
        "placement": getattr(be, "placement", {}),
        "runtime_note": be.runtime_note[:300],
    }


class LayaBackend:
    """convaiinnovations/laya - a different vendor's model in the same category.

    Worth including because the interface is almost identical: a state plus named
    typed questions with runtime criteria, answered in one non-generative forward
    pass, using the same three types and even the same word "noul". Running it on
    the same frozen positions tests whether the findings are about Decision 1.0 or
    about this class of model.
    """

    branch = "laya"

    def __init__(self, name: str = "Laya", device: str = "cuda:0", max_len: int = 4096):
        from laya import Router

        self.name = name
        self.model_id = "laya"
        self.max_len = max_len
        self.router = Router()
        self.routed = {}

    def evaluate(self, request: dict) -> dict:
        try:
            res = self.router.predict(request["state"], request["questions"], max_len=self.max_len)
        except TypeError:          # older runtimes without max_len
            res = self.router.predict(request["state"], request["questions"])
        which = (res.get("routing") or {}).get("model")
        self.routed[which] = self.routed.get(which, 0) + 1
        return {"model": "laya", "answers": res["answers"],
                "usage": {"input_tokens": None}}


def open_backend(name: str, device: str = "cuda:0", max_memory: dict | None = None):
    """Load whichever branch `name` belongs to."""
    if name.lower().startswith("laya"):
        return LayaBackend(name)
    if name in NATIVE:
        return EncoderBackend(name, device=device)
    return DecoderBackend(name, device=device, max_memory=max_memory)


def build_request(model_id: str, state: str, questions: dict) -> dict:
    """A System One request is exactly three keys - extras are a hard ValueError."""
    return {"model": model_id, "state": state, "questions": questions}


def validate_device_agreement(name: str = "Kai") -> dict:
    """Run the vendor's own packaged example on CPU and CUDA and diff the probabilities."""
    _install_shim()
    root = MODELS / NATIVE[name]["dir"]
    request = json.loads((root / "examples" / "system-one.json").read_text())
    request["model"] = f"Decision-1.0-{name}"

    out = {}
    for device in ("cpu", "cuda:0"):
        be = EncoderBackend(name, device=device)
        out[device] = be.evaluate(request)
        del be

    worst = 0.0
    fields = []
    for qid, cpu_ans in out["cpu"]["answers"].items():
        gpu_ans = out["cuda:0"]["answers"][qid]
        if cpu_ans["type"] == "noul":
            d = abs(cpu_ans["noul"] - gpu_ans["noul"])
            fields.append((f"{qid}.noul", d))
            worst = max(worst, d)
        else:
            for k, v in cpu_ans["probabilities"].items():
                d = abs(v - gpu_ans["probabilities"][k])
                fields.append((f"{qid}.p[{k}]", d))
                worst = max(worst, d)
            if cpu_ans["type"] == "choice":
                assert cpu_ans["choice"] == gpu_ans["choice"], "argmax disagrees across devices"
    return {
        "model": name,
        "max_abs_prob_delta": worst,
        "n_compared": len(fields),
        "argmax_stable": True,
        "cpu": out["cpu"],
        "cuda": out["cuda:0"],
    }


if __name__ == "__main__":
    import time

    rep = validate_device_agreement("Kai")
    print(f"CPU vs CUDA agreement on the vendor example, {rep['n_compared']} probabilities compared")
    print(f"  max |delta p| = {rep['max_abs_prob_delta']:.3e}   argmax stable: {rep['argmax_stable']}")

    be = EncoderBackend("Kai", device="cuda:0")
    req = json.loads((MODELS / "Kai-0.6B" / "examples" / "system-one.json").read_text())
    req["model"] = "Decision-1.0-Kai"
    be.evaluate(req)  # warm up
    t0 = time.time()
    n = 20
    for _ in range(n):
        be.evaluate(req)
    dt = (time.time() - t0) / n
    print(f"  steady-state: {dt*1000:.0f} ms per 3-question request ({3/dt:.0f} decisions/s)")
