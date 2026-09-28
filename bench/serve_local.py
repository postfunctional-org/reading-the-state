"""Serve one in-process answerer (Decision 1.0, Laya, heuristic) as a loopback
SystemOne endpoint, so a driver in another virtualenv can call it with exactly
the request HttpAnswerer sends every hosted model.

The request's single Choice question is turned back into the position the
answerer's own pick() expects - state, instructions, candidates {id: desc} - which
is the inverse of answerers.as_questions(), so the model sees what it would
in-process.

    .venv/bin/python bench/serve_local.py --answerer Nox --port 8801
"""
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answerers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerer", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    a = answerers.build(args.answerer)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            (qid, q), = body["questions"].items()
            pos = {"env": body.get("env", "textworld"), "pid": "live", "state": body["state"],
                   "instructions": q["instructions"],
                   "candidates": [{"id": k, "desc": v, "kind": "other"} for k, v in q["criteria"].items()]}
            choice, info = a.pick(pos)
            out = {"model": a.name, "answers": {qid: {"type": "choice", "choice": choice,
                   "confidence": info.get("confidence"), "probabilities": info.get("probs")}}}
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    print(f"[serve_local] {a.name} on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
