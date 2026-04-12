"""
server.py — HTTP ping server for OpenEnv validator.
Port 7860 — required for HuggingFace Spaces Docker SDK.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[server] {self.command} {self.path}", flush=True)

    def _send_json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send_json(200, {"status": "ok", "env": "inventops", "path": self.path})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._send_json(200, {"status": "ok", "message": "InventOps environment ready"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()


if __name__ == "__main__":
    port = 7860
    print(f"[server] Starting on 0.0.0.0:{port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
