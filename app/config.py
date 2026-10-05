"""Env loading + runtime config, shared by every entry point.

Why this file exists
--------------------
`setup-lite.sh` writes `.env` and `setup-docker.sh` *edits* it (it rewrites
`QDRANT_MODE=server`, `QDRANT_URL=...`, `EMBEDDING_BACKEND=bge-m3` with sed).
Before this module, nothing ever read that file, so the docker path silently
ran the lite path: `Searcher` connected to an in-memory Qdrant instead of the
server container, and every embedding came out 384-d instead of 1024-d. The
result still *worked* -- it just measured the wrong system, which is worse than
a crash for a latency lab.

So: load `.env` once, at import time, from the modules that branch on it
(`app/search.py`, `app/embeddings.py`). Rule is `setdefault`, so a variable
already exported in the shell always wins over the file. That keeps
`EMBEDDING_BACKEND=bge-m3 make benchmark` working as an override, and keeps
pytest's `monkeypatch.setenv` authoritative in tests.

No new dependency on purpose: this is 20 lines of parsing, and `python-dotenv`
is not in requirements.txt for the lite path. Feast ships it transitively but
the rest of `app/` must not depend on that.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def _strip_inline_comment(value: str) -> str:
    """Drop the trailing `# ...` that .env.example puts after each setting.

    `QDRANT_MODE=memory   # memory | server` is a comment for humans, but a
    naive split("=") would make the value `memory   # memory | server`, and
    `os.getenv("QDRANT_MODE") == "server"` would never match again. Only a `#`
    preceded by whitespace counts, so a value may legitimately contain `#`.
    """
    out: list[str] = []
    quote: str | None = None
    for i, ch in enumerate(value):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "#" and i > 0 and value[i - 1].isspace():
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Merge `.env` into `os.environ` without clobbering existing values.

    Returns the mapping that was applied, for logging/debugging. Missing file is
    not an error -- the lab works fine with pure defaults, and `os.environ` may
    legitimately carry everything (e.g. a CI runner exporting QDRANT_URL).
    """
    env_path = Path(path) if path is not None else ENV_PATH
    if not env_path.exists():
        return {}

    applied: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = _strip_inline_comment(value.strip()).strip("'\"")
        if not key:
            continue
        # setdefault, not assign: the environment always beats the file.
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


# Import side effect: every `from app.config import load_dotenv` also loads .env.
load_dotenv()
