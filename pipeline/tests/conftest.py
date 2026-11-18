import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _away_from_the_pipeline_s_data(tmp_path, monkeypatch):
    """Every path of the pipeline's commands into its data or the site's, moved into the
    test's own folder: no test writes (or reads) the real ones."""
    from flaneur import cli

    roots = {cli.DATA: tmp_path / "data", cli.WEB_DATA: tmp_path / "site"}

    def away(value):
        if isinstance(value, tuple):
            return tuple(away(v) for v in value)
        if isinstance(value, Path):
            for root, instead in roots.items():
                if value.is_relative_to(root):
                    return instead / value.relative_to(root)
        return value

    for name, value in list(vars(cli).items()):
        if isinstance(value, Path | tuple) and (moved := away(value)) != value:
            monkeypatch.setattr(cli, name, moved)


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
