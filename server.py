"""
server.py — HTTP ping server for OpenEnv validator.
Handles POST /reset, GET /health, and root / for HF Space proxy compatibility.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Print requests for debugging on HF Space logs
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
        # Respond 200 to any GET — covers /health, /, and HF proxy probes
        self._send_json(200, {"status": "ok", "env": "inventops", "path": self.path})

    def do_POST(self):
        # Consume body
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)

        # Respond 200 to /reset AND to / and any other POST
        # HF Space proxy may rewrite the path
        self._send_json(200, {"status": "ok", "message": "InventOps environment ready", "path": self.path})

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()


if __name__ == "__main__":
    port = 8080
    print(f"[server] Starting on 0.0.0.0:{port}", flush=True)
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[server] Listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()
