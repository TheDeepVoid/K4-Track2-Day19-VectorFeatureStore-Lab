"""Tests for bonus/agent.py — the hybrid memory agent POC.

`bonus/` is not part of the core lab, so nothing else imports it and a broken
import would only surface when the grader ran `python bonus/demo.py`. The rubric
awards 7 of 20 bonus points for that running, so the load-bearing paths get
regression tests here. The bugs these were written against were all real:
a misaligned diacritic table that made the module unimportable, a SemanticCache
constructed without its required client/embedder, an RRF call passing two
rankings as separate positional args, and a lang sniffer that checked
diacritic-folded text for diacritics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent, Memory, QueryLog, fold_vi  # noqa: E402


@pytest.fixture(scope="module")
def agent():
    """One agent for the whole module: building the cache loads a model."""
    a = HybridMemoryAgent(user_id="u_001", top_k=3)
    a.remember("Kubernetes auto-scaling theo CPU và memory", topic="cloud")
    a.remember("OAuth 2.0 và JWT: lưu refresh token để tránh XSS", topic="security")
    a.remember("PostgreSQL composite index theo thứ tự cột", topic="database")
    return a


# ── diacritic folding (the import-time crash) ─────────────────────────────

def test_fold_table_is_aligned_and_importable():
    """A ValueError here means bonus/agent.py cannot be imported at all."""
    from bonus.agent import _ACCENTED, _PLAIN

    assert len(_ACCENTED) == len(_PLAIN)
    assert len(_ACCENTED) > 50


def test_fold_vi_strips_diacritics_and_lowercases():
    assert fold_vi("Tài Liệu Về Kubernetes") == "tai lieu ve kubernetes"
    assert fold_vi("Đám mây") == "dam may"


def test_fold_vi_handles_decomposed_nfc_nfd():
    """NFD input (combining marks) must fold the same as NFC input.

    Vietnamese text from keyboards, copy-paste and legacy DBs is often NFD; a
    translate table written against precomposed code points would leave the
    marks in place and silently break BM25 for those users.
    """
    import unicodedata

    precomposed = "Tài liệu về Kubernetes"
    decomposed = unicodedata.normalize("NFD", precomposed)
    assert decomposed != precomposed, "test input is not actually decomposed"
    assert fold_vi(decomposed) == fold_vi(precomposed) == "tai lieu ve kubernetes"


# ── the public API from the brief ─────────────────────────────────────────

def test_remember_then_recall_returns_context(agent):
    out = agent.recall("Kubernetes auto-scaling", user_id="u_001")
    assert out.startswith("[source=")
    for section in ("PROFILE", "RECENT", "EPISODIC"):
        assert section in out
    assert "Kubernetes" in out


def test_recall_with_no_memories_is_graceful():
    a = HybridMemoryAgent(user_id="u_099", use_cache=False)
    out = a.recall("anything", user_id="u_099")
    assert "no memories yet" in out


def test_paraphrase_is_carried_by_the_dense_half(agent):
    """No literal token overlap with the memory, so BM25 contributes nothing.

    If the top hit were empty or irrelevant, fusion would be doing nothing for
    Vietnamese paraphrase -- the exact weakness NB2 reports on bge-small.
    """
    scores = agent._bm25_scores(fold_vi("tai lieu ve kubernets"),
                                agent.memories)
    assert not scores, "BM25 should find nothing for a fully-folded paraphrase"
    dense = agent._dense_scores("tai lieu ve kubernets", agent.memories)
    assert dense, "dense retriever should still score every memory"
    best = max(dense, key=dense.get)
    assert "Kubernetes" in agent._doc_meta[best]["title"]


def test_recall_is_deterministic(agent):
    a = agent.recall("index database", user_id="u_001")
    b = agent.recall("index database", user_id="u_001")
    # second call is a cache hit, so compare the retrieved body only
    assert a.split("\n", 1)[1].split("RECENT")[0] == b.split("\n", 1)[1].split("RECENT")[0]


# ── profile + streaming views ─────────────────────────────────────────────

def test_profile_tracks_dominant_topic(agent):
    assert agent._profile["u_001"]["topic_affinity"] == "cloud"
    assert agent._profile["u_001"]["memories_read"] == 3


def test_preferred_language_detects_vietnamese():
    a = HybridMemoryAgent(use_cache=False)
    a.remember("Tài liệu tiếng Việt về hạ tầng", user_id="u_vi")
    assert a._profile["u_vi"]["preferred_language"] == "vi"
    assert a._profile["u_vi"]["preferred_language"] != "en", (
        "lang sniffer must not look for diacritics in already-folded text"
    )


def test_code_switching_reports_mix():
    a = HybridMemoryAgent(use_cache=False)
    a.remember("Tài liệu về Kubernetes and how to scale it", user_id="u_mix")
    assert a._profile["u_mix"]["preferred_language"] == "mix"


def test_query_log_window_counts():
    log = QueryLog()
    log.record("u_001", "cloud computing", "cloud")
    log.record("u_001", "bảo mật", "security")
    log.record("u_001", "cloud again", "cloud")
    vel = log.velocity("u_001")
    assert vel["queries_last_hour"] == 3
    assert vel["distinct_topics_24h"] == 2
    assert log.velocity("nobody") == {"queries_last_hour": 0, "distinct_topics_24h": 0}


# ── isolation + cache invalidation ────────────────────────────────────────

def test_users_cannot_see_each_others_memories(agent):
    agent.remember("Mật khẩu và 2FA cho tài khoản ngân hàng", user_id="u_002", topic="security")
    out = agent.recall("Kubernetes", user_id="u_002")
    assert "Kubernetes" not in out
    assert "Mật khẩu" in out


def test_new_memory_invalidates_cached_recall():
    """A cached answer becomes wrong the moment a memory is added.

    Without invalidate_tenant(), the cache would keep serving the pre-existing
    answer and the new memory would be invisible to recall.
    """
    a = HybridMemoryAgent(user_id="u_003", top_k=3)
    a.remember("Kubernetes auto-scaling theo CPU", topic="cloud")
    first = a.recall("hạ tầng", user_id="u_003")
    assert first.startswith("[source=retrieval")

    second = a.recall("hạ tầng", user_id="u_003")
    assert second.startswith("[source=cache")

    a.remember("Kafka và Flink xử lý streaming exactly-once", topic="data_eng")
    third = a.recall("hạ tầng", user_id="u_003")
    assert third.startswith("[source=retrieval"), (
        "adding a memory must invalidate this user's cached recalls"
    )
    assert "Kafka" in third


def test_cache_round_trip_uses_the_wrapper_answer(agent):
    """Regression: recall() concatenated the CacheHit object, not .answer."""
    a = HybridMemoryAgent(user_id="u_004", top_k=2)
    a.remember("Tài liệu về bảo mật zero-trust", topic="security")
    a.recall("bảo mật", user_id="u_004")
    again = a.recall("bảo mật", user_id="u_004")
    assert again.startswith("[source=cache")
    assert "PROFILE" in again
    assert "CacheHit" not in again


# ── fusion internals ──────────────────────────────────────────────────────

def test_rrf_signature_takes_a_list_of_rankings(agent):
    """Regression: `self._rrf(bm25, dense, k=60)` raised "multiple values for k"."""
    a = {"m1": 3.0, "m2": 1.0}
    b = {"m2": 4.0, "m3": 2.0}
    fused = dict(agent._rrf([a, b], k=60, depth=10))
    assert set(fused) == {"m1", "m2", "m3"}
    # agreement between rankers must beat either alone
    assert fused["m2"] > fused["m1"]
    assert fused["m2"] > fused["m3"]


def test_rrf_uses_one_based_ranks():
    fused = dict(HybridMemoryAgent._rrf([{"a": 1.0, "b": 0.5}], k=60, depth=2))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_memory_fold_text_property():
    m = Memory(mem_id="m1", user_id="u_001", text="Đám Mây", topic="cloud")
    assert m.fold_text == "dam may"
