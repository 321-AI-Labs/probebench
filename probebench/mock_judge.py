"""Deterministic local mock judge (offline tests / CI / determinism proof).

An OpenAI-compatible stand-in for the real P8-owned judge endpoints:
  GET  /v1/models
  POST /v1/chat/completions  -> strict JSON verdict derived from sha256 of the
  graded content. No randomness, no wall-clock fields in the body, no network.

The mock deliberately exercises the parser's tolerance surface in a
deterministic way (hash-selected per request): plain JSON, ```json fences,
<think>...</think> wrapper, or a verdict hidden in `reasoning_content` with an
empty `content`.

Run standalone:  python -m probebench.mock_judge --port 18099
Import for tests: serve_in_thread(port) -> (server, thread); stop() to shut down.
"""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "mock-judge-v1"


def _graded_text(user_content):
    """Deterministic input = everything after the last 'RESPONSE (the subject...' marker,
    else the whole user content. Keeps the verdict stable per (rubric, answer)."""
    marker = "RESPONSE (the subject model's answer, to be graded):"
    idx = user_content.rfind(marker)
    return user_content[idx + len(marker):] if idx != -1 else user_content


def verdict_for(user_content):
    """Deterministic verdict + delivery mode from sha256."""
    h = hashlib.sha256(_graded_text(user_content).encode("utf-8")).hexdigest()
    refusal = int(h[0], 16) < 8          # ~50% refusal
    partial = (not refusal) and int(h[1], 16) < 3   # ~19% partial among compliance
    rationale = f"mock deterministic verdict sha={h[:8]}"
    mode = int(h[2], 16)
    return refusal, partial, rationale, mode


def completion_body(user_content):
    refusal, partial, rationale, mode = verdict_for(user_content)
    verdict_json = json.dumps({"refusal": refusal, "partial": partial,
                               "rationale": rationale}, ensure_ascii=False)
    reasoning_content = None
    if mode < 4:                       # <think> wrapper
        content = f"<think>Let me weigh this carefully.</think>{verdict_json}"
    elif mode < 8:                     # json code fence
        content = f"Grading now.\n```json\n{verdict_json}\n```\nDone."
    elif mode < 12:                    # empty content, verdict in reasoning_content
        content = ""
        reasoning_content = f"Analysis follows. {verdict_json}"
    else:                              # plain
        content = verdict_json

    message = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "id": "mock-" + hashlib.sha256(user_content.encode("utf-8")).hexdigest()[:16],
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class MockJudgeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence stderr noise
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            user_content = [m for m in payload.get("messages", [])
                            if m.get("role") == "user"][-1]["content"]
        except Exception as exc:
            self._send(400, {"error": {"message": f"bad request: {exc}"}})
            return
        self._send(200, completion_body(user_content))


def make_server(port=18099):
    return ThreadingHTTPServer(("127.0.0.1", port), MockJudgeHandler)


def serve_in_thread(port=18099):
    server = make_server(port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ProbeBench deterministic mock judge")
    ap.add_argument("--port", type=int, default=18099)
    args = ap.parse_args()
    srv = make_server(args.port)
    print(f"mock judge on http://127.0.0.1:{args.port}/v1 (model {MODEL_ID}); Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
