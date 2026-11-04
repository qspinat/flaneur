import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    return lambda name: json.loads((FIXTURES / name).read_text())


@pytest.fixture
def ask():
    return _ask


def _ask(server, *requests):
    """Send requests to a local page's server, one after the other, and close it: each request
    is (method, path[, body[, headers]]), a dict body sent as JSON; (status, bytes) for each."""
    import http.client
    import threading

    def serve():
        for _ in requests:
            server.handle_request()

    threading.Thread(target=serve, daemon=True).start()
    answers = []
    try:
        for method, path, *rest in requests:
            body, headers = (rest + [None, None])[:2]
            if isinstance(body, dict):
                body, headers = (
                    json.dumps(body),
                    {"Content-Type": "application/json", **(headers or {})},
                )
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            answers.append((response.status, response.read()))
            connection.close()
    finally:
        server.server_close()
    return answers
