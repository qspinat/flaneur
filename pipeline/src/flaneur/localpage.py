"""What the local pages (review, landmarks, confirmation) share: a quiet handler answering in
bytes or JSON, listening on localhost only. Another website open in the browser can still send
requests there: a request naming another host (DNS rebinding) is refused, and a body is read
only when it is sent as JSON, which a page of another origin cannot do without asking first
(a CORS preflight, never answered here)."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ZOOM = Path(__file__).with_name("zoom.js")
INCLUDE = '<script src="zoom.js"></script>'  # where a page takes the shared zoom


def page(path: Path) -> bytes:
    """A local page as served: the shared zoom put in where it asks for it."""
    text = path.read_text(encoding="utf-8")
    if INCLUDE not in text:
        raise ValueError(f"{path.name} ne demande pas le zoom partagé ({INCLUDE})")
    zoom = ZOOM.read_text(encoding="utf-8")
    return text.replace(INCLUDE, f"<script>\n{zoom}</script>").encode()


class Handler(BaseHTTPRequestHandler):
    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        port = self.server.server_address[1]
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._send(403, b"forbidden", "text/plain")
            return False
        return True

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value) -> None:
        self._send(200, json.dumps(value, ensure_ascii=False).encode(), "application/json")

    def _body(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("le corps doit être du JSON (Content-Type: application/json)")
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def log_message(self, *args) -> None:  # quiet
        pass


def server(handler: type[Handler], port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), handler)
