"""
server.py — Lightweight HTTP server for OpenEnv validator ping.

Endpoints:
    POST /reset  → 200  (validator ping)
    GET  /health → 200  (Docker healthcheck)
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Suppress default access logs to keep stdout clean
        pass

    def _send_json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "env": "inventops"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/reset":
            # Consume request body if present
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            self._send_json(200, {"status": "ok", "message": "InventOps environment ready"})
        else:
            self._send_json(404, {"error": "not found"})


if __name__ == "__main__":
    port = 8080
    print(f"[server] Listening on 0.0.0.0:{port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
