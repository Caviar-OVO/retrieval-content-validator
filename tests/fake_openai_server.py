"""A tiny OpenAI-compatible stub server to exercise the real Ragas/TruLens
code paths without a paid API key.

Run with:  python tests/fake_openai_server.py
Then:      $env:OPENAI_API_KEY="test"; $env:OPENAI_BASE_URL="http://127.0.0.1:8765/v1"
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode() if length else "{}"
        try:
            req = json.loads(raw)
        except Exception:
            req = {}

        # Ragas/TruLens use JSON-mode chat completions. We return a generic
        # payload; the aim is only to prove the transport + parsing path runs.
        content = json.dumps({
            "statements": ["stub statement"],
            "reason": "stub",
            "verdict": 1,
            "score": 1,
        })
        self._send({
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": req.get("model", "stub"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def do_GET(self):
        self._send({"data": [{"id": "stub", "object": "model"}]})


def main():
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()


if __name__ == "__main__":
    main()