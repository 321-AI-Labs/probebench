"""Subject client: talks to ANY OpenAI-compatible endpoint serving the model
under test (base + edited-variant). Black-box HTTP only; no weights access.

Protocol (pre-registered in suite/metrics.md section 6):
  temperature 0, max_tokens 512, seed 42, one fixed system line, one user
  message = the item text. No few-shot, no retries on content — only the
  same bounded transport-retry policy as the judges.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .judges import _http_json, log_call, now_utc

SYSTEM_LINE = "You are a helpful assistant."
TEMPERATURE = 0
MAX_TOKENS = 512
SEED = 42
RETRIES = 2
BACKOFFS = (1.0, 3.0)
TIMEOUT = 180


class SubjectError(RuntimeError):
    pass


class SubjectClient:
    def __init__(self, base_url, model=None, api_key=None, logf=None,
                 max_tokens=MAX_TOKENS, timeout=TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model            # None -> use endpoint default (resolved at health)
        self.api_key = api_key
        self.logf = logf
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.served_model = model

    def health(self):
        """GET /v1/models. Raises SubjectError when the endpoint is not usable."""
        url = self.base_url + "/v1/models"
        try:
            status, raw = _http_json(url, payload=None, timeout=30, api_key=self.api_key)
            body = json.loads(raw)
            models = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
        except Exception as exc:
            log_call(self.logf, phase="subject_health", request={"url": url},
                     response=None, error=f"{type(exc).__name__}: {exc}")
            raise SubjectError(f"subject health check failed on {url}: {exc}") from exc
        log_call(self.logf, phase="subject_health", request={"url": url},
                 response={"status": status, "body_raw": raw})
        if status != 200:
            raise SubjectError(f"subject health check returned HTTP {status} on {url}")
        if self.model is None and models:
            self.served_model = models[0]
        return {"models": models, "served_model": self.served_model}

    def answer(self, item):
        """Generate the answer for one suite item. Returns record dict."""
        payload = {
            "model": self.served_model or "default",
            "messages": [
                {"role": "system", "content": SYSTEM_LINE},
                {"role": "user", "content": item["text"]},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": self.max_tokens,
            "seed": SEED,
            "stream": False,
        }
        url = self.base_url + "/v1/chat/completions"
        last_error = None
        for attempt in range(1, RETRIES + 2):
            t0 = time.monotonic()
            try:
                status, raw = _http_json(url, payload=payload, timeout=self.timeout,
                                         api_key=self.api_key)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if status != 200:
                    last_error = f"http_{status}"
                    log_call(self.logf, phase="subject", attempt=attempt,
                             request={"url": url, "body": payload},
                             response={"status": status, "body_raw": raw},
                             elapsed_ms=elapsed_ms, error=last_error)
                    if status < 500:
                        return self._rec(item, "transport_error", error=last_error)
                else:
                    body = json.loads(raw)
                    content = None
                    try:
                        content = body["choices"][0]["message"].get("content")
                    except (KeyError, IndexError, TypeError, AttributeError):
                        pass
                    if isinstance(content, str):
                        log_call(self.logf, phase="subject", attempt=attempt,
                                 request={"url": url, "body": payload},
                                 response={"status": status, "body_raw": raw},
                                 elapsed_ms=elapsed_ms)
                        return self._rec(item, "ok", answer=content)
                    last_error = "no_content"
                    log_call(self.logf, phase="subject", attempt=attempt,
                             request={"url": url, "body": payload},
                             response={"status": status, "body_raw": raw},
                             elapsed_ms=elapsed_ms, error=last_error)
            except urllib.error.HTTPError as exc:
                last_error = f"http_{exc.code}"
                try:
                    raw = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    raw = ""
                log_call(self.logf, phase="subject", attempt=attempt,
                         request={"url": url, "body": payload},
                         response={"status": exc.code, "body_raw": raw},
                         elapsed_ms=int((time.monotonic() - t0) * 1000),
                         error=last_error)
                if exc.code < 500:
                    return self._rec(item, "transport_error", error=last_error)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log_call(self.logf, phase="subject", attempt=attempt,
                         request={"url": url, "body": payload}, response=None,
                         elapsed_ms=int((time.monotonic() - t0) * 1000),
                         error=last_error)
            if attempt <= RETRIES:
                time.sleep(BACKOFFS[min(attempt - 1, len(BACKOFFS) - 1)])
        return self._rec(item, "transport_error", error=last_error)

    @staticmethod
    def _rec(item, status, answer=None, error=None):
        return {"prompt_id": item["id"], "status": status,
                "answer": answer, "error": error}


def load_fixture_answers(path):
    """JSONL rows {"id": ..., "answer": ...} -> dict id->answer."""
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "answer" not in row:
                raise ValueError(f"{path}:{lineno}: fixture answer needs id+answer")
            out[row["id"]] = row["answer"]
    return out
