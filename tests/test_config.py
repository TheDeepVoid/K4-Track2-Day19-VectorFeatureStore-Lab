"""Tests for app/config.py — the .env loader that decides which stack runs.

These guard the bug that made the Docker path silently measure the lite path:
`setup-docker.sh` rewrites `.env` to `QDRANT_MODE=server`, but nothing read the
file, so Qdrant stayed in-memory and the "bge-m3 1024-d" run was actually
bge-small 384-d. The numbers still looked plausible, which is the dangerous part.
"""
from __future__ import annotations

import os

import pytest

from app.config import _strip_inline_comment, load_dotenv


def write_env(tmp_path, text: str):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Importing app.config loads the real repo .env.

    Every test here writes its own file and asserts on what the loader applied,
    so the real values must not already be sitting in os.environ.
    """
    for key in ("QDRANT_MODE", "QDRANT_URL", "EMBEDDING_BACKEND", "A", "B"):
        monkeypatch.delenv(key, raising=False)


def test_loads_key_value_pairs(tmp_path):
    env = write_env(tmp_path, "QDRANT_MODE=server\nEMBEDDING_BACKEND=bge-m3\n")
    applied = load_dotenv(env)
    assert applied == {"QDRANT_MODE": "server", "EMBEDDING_BACKEND": "bge-m3"}
    assert os.environ["QDRANT_MODE"] == "server"
    assert os.environ["EMBEDDING_BACKEND"] == "bge-m3"


def test_existing_environment_wins_over_file(tmp_path, monkeypatch):
    """setdefault, not assign: `EMBEDDING_BACKEND=bge-m3 make benchmark` must work."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    env = write_env(tmp_path, "EMBEDDING_BACKEND=fastembed\n")
    load_dotenv(env)
    assert os.environ["EMBEDDING_BACKEND"] == "openai"


def test_inline_comments_are_stripped(tmp_path):
    """.env.example writes `QDRANT_MODE=memory   # memory | server`.

    Without stripping, the value becomes "memory   # memory | server" and the
    `== "server"` comparison never matches again.
    """
    env = write_env(tmp_path, "QDRANT_MODE=server   # memory | server\n")
    load_dotenv(env)
    assert os.environ["QDRANT_MODE"] == "server"


def test_comments_blanks_and_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    env = write_env(
        tmp_path,
        "# leading comment\n\nA='quoted value'\nB=\"double quoted\"  # trailing\nbad line\n",
    )
    applied = load_dotenv(env)
    assert applied == {"A": "quoted value", "B": "double quoted"}
    assert "bad line" not in applied


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_hash_inside_value_survives():
    """A `#` only starts a comment when preceded by whitespace."""
    assert _strip_inline_comment("v1.2#frag") == "v1.2#frag"
    assert _strip_inline_comment("value # comment") == "value"


def test_app_search_builds_server_client_when_env_says_so(monkeypatch):
    """The end-to-end claim: QDRANT_MODE=server really selects the server client.

    Guards the wiring, not just the parser. A loader that nothing calls would
    pass every test above and still be inert -- which is exactly the bug this
    module exists to fix.
    """
    import app.search as search_mod

    monkeypatch.setenv("QDRANT_MODE", "server")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    constructed: list[dict] = []

    class FakeClient:
        def __init__(self, *a, **kw):
            constructed.append({"args": a, "kwargs": kw})

        def get_collections(self):
            return type("R", (), {"collections": []})()

        def create_collection(self, **kw):
            pass

        def upsert(self, **kw):
            pass

    monkeypatch.setattr(search_mod, "QdrantClient", FakeClient)

    s = search_mod.Searcher()
    s.docs = [{"doc_id": "cloud_000", "title": "t", "text": "body"}]
    s._build_vector_index()

    assert constructed, "no QdrantClient was constructed"
    assert constructed[0]["kwargs"] == {"url": "http://localhost:6333"}


def test_app_search_defaults_to_memory(monkeypatch):
    """No env var at all must stay on the in-memory path (the lite default)."""
    import app.search as search_mod

    constructed: list[tuple] = []

    class FakeClient:
        def __init__(self, *a, **kw):
            constructed.append(a)

        def get_collections(self):
            return type("R", (), {"collections": []})()

        def create_collection(self, **kw):
            pass

        def upsert(self, **kw):
            pass

    monkeypatch.setattr(search_mod, "QdrantClient", FakeClient)

    s = search_mod.Searcher()
    s.docs = [{"doc_id": "cloud_000", "title": "t", "text": "body"}]
    s._build_vector_index()

    assert constructed[0] == (":memory:",)
