"""Bonus challenge — HybridMemoryAgent (Vietnamese-first personal AI memory POC).

Combines the three memory types from the brief:
  * episodic  -> vector store   (app/search.py: BM25 + vector + RRF hybrid)
  * profile   -> feature store  (Feast online features, TTL-based freshness)
  * recent    -> streaming-ish  (app/cache.py semantic cache + velocity counters)

Design decisions are documented in bonus/ARCHITECTURE.md. The code here is
deliberately clarity-first (per the brief: "optimize clarity, not speed").

Lab concepts reused on purpose:
  NB2 RRF fusion          -> HybridMemoryAgent.recall() fuses BM25 + vector via RRF
  NB4 Feast online store  -> profile features (topic_affinity, preferred_language)
  NB5 filtered-ANN        -> every episodic memory carries a user_id payload filter,
                             so user A can never retrieve user B's memories
  NB7 semantic cache      -> near-duplicate recalls are served from cache (TTL-bound)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdrant_client import QdrantClient  # noqa: E402

from app.cache import SemanticCache  # noqa: E402
from app.embeddings import Embedder  # noqa: E402

# ── Streaming-flavoured recent-activity tracker ───────────────────────────
# Stands in for the "streaming feature view" of the brief without needing a
# Kafka/Flink pipeline: it is a per-user in-memory sliding window, and it keeps
# exactly the two counters query_velocity_features declares in app/feast_repo.
SLIDING_WINDOW = timedelta(hours=1)

# Vietnamese code-switching: users type "docs ve Kubernetes" (missing diacritics)
# as often as "tài liệu về Kubernetes". Diacritic folding makes the BM25 half of
# the hybrid retriever work for un-accented input without a second model.
#
# Written as (accented, plain) pairs rather than two hand-aligned strings: the
# original table was 74 vs 69 characters and raised ValueError on import, so the
# module could not even be imported. Grouping by base letter keeps the two sides
# provably aligned, and the assert below fails loudly if anyone edits it wrong.
_VI_FOLD_PAIRS: list[tuple[str, str]] = [
    ("àáạảãâầấậẩẫăằắặẳẵ", "a"),
    ("èéẹẻẽêềếệểễ", "e"),
    ("ìíịỉĩ", "i"),
    ("òóọỏõôồốộổỗơờớợởỡ", "o"),
    ("ùúụủũưừứựửữ", "u"),
    ("ỳýỵỷỹ", "y"),
    ("đ", "d"),
]
_ACCENTED = "".join(acc for acc, _ in _VI_FOLD_PAIRS)
_PLAIN = "".join(plain * len(acc) for acc, plain in _VI_FOLD_PAIRS)
assert len(_ACCENTED) == len(_PLAIN), "diacritic fold table is misaligned"
_FOLD = str.maketrans(_ACCENTED, _PLAIN)


def fold_vi(text: str) -> str:
    """Strip diacritics + lowercase — cheap Vietnamese/English code-switch fix.

    NFC-normalises first: Vietnamese text arriving from keyboards, copy-paste or
    legacy DBs is often NFD (combining marks), and a translate table written
    against precomposed code points would silently leave those marks in place.
    """
    return unicodedata.normalize("NFC", text).lower().translate(_FOLD)


@dataclass
class Memory:
    """One episodic memory chunk (one document the user saved/read)."""

    mem_id: str
    user_id: str
    text: str
    topic: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # what a user "reads" — cheap, deterministic, enough for the POC
    dwell_seconds: int = 45

    @property
    def fold_text(self) -> str:
        return fold_vi(self.text)


class QueryLog:
    """Sliding-window recent-activity tracker (the streaming feature view).

    Feeds `queries_last_hour` and `distinct_topics_24h` — the same two features
    as Feast's `query_velocity_features` (ttl=1h), just computed live.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

    def record(self, user_id: str, query: str, topic: str | None) -> None:
        now = datetime.now(timezone.utc)
        evts = self._events[user_id]
        evts.append((now, topic or "unknown"))
        # drop anything outside the window — this IS the TTL
        evts[:] = [(t, tp) for t, tp in evts if now - t <= SLIDING_WINDOW]

    def velocity(self, user_id: str) -> dict:
        evts = self._events.get(user_id, [])
        topics = [tp for _, tp in evts]
        return {
            "queries_last_hour": len(evts),
            "distinct_topics_24h": len(set(topics)),
        }


class HybridMemoryAgent:
    """Personal AI memory: episodic (vector) + profile (features) + recent (stream).

    Public API is the 2 methods the brief requires:
        remember(text, user_id)  -> None
        recall(query, user_id)   -> str
    """

    def __init__(self, user_id: str = "u_001", top_k: int = 5,
                 use_profile: bool = True, use_cache: bool = True) -> None:
        # The brief specifies `remember(text, user_id="u_001")`, which is right
        # for the default agent but wrong for any other: constructing with
        # user_id="u_002" and then calling remember(text) would file the memory
        # under u_001 and every recall would come back empty. Both methods
        # therefore default to *this agent's* user, which is identical to
        # "u_001" for the default agent and correct for the rest.
        self.user_id = user_id
        self.top_k = top_k
        self.use_profile = use_profile

        # NB7 lesson: namespace by user_id so a shared cache cannot leak one
        # user's memory into another's recall (namespaced=True is the safe default).
        # SemanticCache owns a Qdrant collection + an embedder, so both are
        # required args -- and dim must follow the embedder, not the 384 default,
        # or a non-default EMBEDDING_BACKEND silently mismatches on insert.
        self._embedder = Embedder()
        self.cache: SemanticCache | None = None
        if use_cache:
            self.cache = SemanticCache(
                client=QdrantClient(":memory:"),
                embedder=self._embedder,
                dim=self._embedder.dim,
                threshold=0.80,
                ttl_s=300,
                namespaced=True,
            )

        self.query_log = QueryLog()

        # Episodic memory starts empty and grows with `remember()`.
        self.memories: list[Memory] = []
        self._doc_meta: dict[str, dict] = {}

        # Stable profile — "offline" features, slow-moving (NB4 TTL=30d).
        self._profile: dict[str, dict] = defaultdict(
            lambda: {"topic_affinity": None, "preferred_language": "vi",
                     "reading_speed_wpm": 220, "memories_read": 0}
        )
        self._lang_hist: dict[str, set[str]] = defaultdict(set)

    # ── writers ──────────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str | None = None,
                 topic: str | None = None) -> None:
        """Add a new piece of episodic memory for this user (Lab §1 vector store)."""
        user_id = user_id or self.user_id
        topic = topic or self._sniff_topic(text)
        mem = Memory(mem_id=f"{user_id}_mem_{len(self.memories):04d}",
                     user_id=user_id, text=text, topic=topic)
        self.memories.append(mem)
        self._doc_meta[mem.mem_id] = {"doc_id": mem.mem_id, "topic": topic,
                                      "title": text[:60], "user_id": user_id}
        prof = self._profile[user_id]
        prof["memories_read"] += 1
        self._lang_hist[user_id].add(self._sniff_lang(text))
        # keep the profile's affinity pointing at the current dominant topic
        counts = defaultdict(int)
        for m in self.memories:
            if m.user_id == user_id:
                counts[m.topic] += 1
        prof["topic_affinity"] = max(counts, key=counts.get) if counts else None
        # _lang_hist stores per-memory labels, and a code-switched memory is
        # labelled "mix" rather than counted as both -- so the profile can be
        # "mix" from a single bilingual memory, and must not need two of them.
        seen = self._lang_hist[user_id]
        if "mix" in seen or ("vi" in seen and "en" in seen):
            prof["preferred_language"] = "mix"
        elif "en" in seen:
            prof["preferred_language"] = "en"
        else:
            prof["preferred_language"] = "vi"
        # a new memory changes the answer for every question this user might ask,
        # so drop their cached recalls. Without this the semantic cache happily
        # serves a pre-existing answer forever and the new memory is invisible.
        #
        # Implemented here rather than as a SemanticCache method: app/cache.py is
        # part of the lab's own source and is not ours to extend. The cache
        # exposes its client and collection, so the scroll+delete is reachable
        # without touching it.
        if self.cache is not None:
            self._invalidate_cached_recalls(user_id)

    def _invalidate_cached_recalls(self, user_id: str) -> int:
        """Drop every cached answer for one user. Returns how many were removed."""
        from qdrant_client import models

        from app.cache import CACHE_COLLECTION

        qf = models.Filter(must=[models.FieldCondition(
            key="tenant", match=models.MatchValue(value=user_id))])
        doomed = [p.id for p in self.cache.client.scroll(
            collection_name=CACHE_COLLECTION,
            scroll_filter=qf,
            limit=10_000,
            with_payload=False,
        )[0]]
        if doomed:
            self.cache.client.delete(
                collection_name=CACHE_COLLECTION,
                points_selector=models.PointIdsList(points=doomed),
            )
        return len(doomed)

    # ── readers ──────────────────────────────────────────────────────────
    def recall(self, query: str, user_id: str | None = None) -> str:
        """Retrieve top-K memories + profile features -> assembled LLM context."""
        user_id = user_id or self.user_id
        self.query_log.record(user_id, query, self._sniff_topic(query))

        # NB7: serve near-duplicate recalls from the semantic cache (TTL-bound).
        # get() returns a CacheHit, not a str -- take .answer, not the wrapper.
        hit = self.cache.get(user_id, query) if self.cache is not None else None
        source = "cache"
        if hit is not None:
            body = hit.answer
        else:
            source = "retrieval"
            body = self._retrieve_and_assemble(query, user_id)
            if self.cache is not None:
                self.cache.put(user_id, query, body)
        return f"[source={source} | user={user_id}]\n" + body

    def _retrieve_and_assemble(self, query: str, user_id: str) -> str:
        user_mem = [m for m in self.memories if m.user_id == user_id]
        if not user_mem:
            return " episodic: (no memories yet — call remember() first)"

        affinity = self._profile[user_id]["topic_affinity"] if self.use_profile else None

        # ── Episodic: RRF fusion of BM25 (folded, code-switch-safe) + dense ──
        # Both retrievers run over the SAME candidate set (this user's memories).
        bm25 = self._bm25_scores(fold_vi(query), user_mem)
        dense = self._dense_scores(query, user_mem)
        # NB2 formula, unchanged: 1/(k + rank), rank 1-based, k=60.
        # _rrf takes a *list* of rankings, so both retrievers go in one list.
        fused = self._rrf([bm25, dense], k=60, depth=self.top_k * 4)

        # ── Profile: boost the user's affinity topic (NB6 personalization) ──
        if affinity:
            # feed affinity as a light prior, NOT a hard filter: a hard topic
            # filter can starve a genuinely off-affinity memory (NB5 lesson).
            fused = [
                (mid, score + (0.05 if self._doc_meta[mid]["topic"] == affinity else 0))
                for mid, score in fused
            ]
        top = sorted(fused, key=lambda kv: -kv[1])[: self.top_k]

        vel = self.query_log.velocity(user_id)
        prof = self._profile[user_id]
        lines = [
            "PROFILE (feature store, TTL=30d):",
            f"  topic_affinity={affinity} preferred_language={prof['preferred_language']}"
            f" reading_speed_wpm={prof['reading_speed_wpm']} memories_read={prof['memories_read']}",
            f"RECENT (streaming, window=1h): queries_last_hour={vel['queries_last_hour']}"
            f" distinct_topics_24h={vel['distinct_topics_24h']}",
            "EPISODIC (vector store, RRF bm25+dense):",
        ]
        for mid, score in top:
            m = self._doc_meta[mid]
            lines.append(f"  - [{m['topic']}] score={score:.4f} {m['title']}")
        if not top:
            lines.append("  (none above threshold)")
        return "\n".join(lines)

    # ── retrievers (small, explicit — clarity over speed) ────────────────
    def _bm25_scores(self, folded_query: str, mems: list[Memory]) -> dict[str, float]:
        q_terms = [t for t in re.findall(r"\w+", folded_query) if t]
        if not q_terms:
            return {}
        scores: dict[str, float] = {}
        for m in mems:
            doc_terms = set(re.findall(r"\w+", m.fold_text))
            overlap = sum(1 for t in q_terms if t in doc_terms)
            if overlap:
                scores[m.mem_id] = overlap / len(q_terms)
        return scores

    def _dense_scores(self, query: str, mems: list[Memory]) -> dict[str, float]:
        """Cosine similarity of the query against every memory.

        Uses the shared `Embedder` directly rather than `Searcher`: Searcher
        owns a *corpus* index built once from data/corpus_vn.jsonl, whereas
        episodic memory here is a small, growing, per-user list. Re-indexing
        1000 corpus docs to score 12 memories would be absurd, so the POC does
        the cosine math in-process over exactly the candidate set. Same vectors,
        same model, same NB2 semantics -- just no corpus in the way.
        """
        import numpy as np

        texts = [m.text for m in mems]
        vecs = np.asarray(list(self._embedder.embed([query] + texts)), dtype=np.float32)
        qv, mv = vecs[0], vecs[1:]
        qn = qv / np.linalg.norm(qv)
        mn = mv / np.linalg.norm(mv, axis=1, keepdims=True)
        return {m.mem_id: float(s) for m, s in zip(mems, mn @ qn)}

    @staticmethod
    def _rrf(rankings: list[dict[str, float]], k: int = 60,
             depth: int = 10) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion, 1/(k+rank), rank 1-based (NB2 formula)."""
        fused: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, (mid, _) in enumerate(
                    sorted(ranking.items(), key=lambda kv: -kv[1])[:depth], start=1):
                fused[mid] += 1.0 / (k + rank)
        return list(fused.items())

    # ── tiny helpers ─────────────────────────────────────────────────────
    TOPIC_HINTS = {
        "cloud": ["cloud", "đám mây", "kubernetes", "serverless", "aws", "mở rộng"],
        "ai_ml": ["ai", "mô hình", "embedding", "llm", "học sâu"],
        "security": ["bảo mật", "security", "oauth", "jwt", "mã hoá"],
        "database": ["cơ sở dữ liệu", "database", "sql", "b-tree"],
        "networking": ["mạng", "network", "tcp", "dns"],
        "devops": ["devops", "ci/cd", "docker", "triển khai"],
        "mobile": ["mobile", "android", "ios"],
        "frontend": ["frontend", "giao diện", "react", "css"],
        "backend": ["backend", "api", "microservice"],
        "data_eng": ["kafka", "flink", "streaming", "etl"],
    }

    @classmethod
    def _sniff_topic(cls, text: str) -> str:
        low = fold_vi(text)
        best, best_n = "general", 0
        for topic, hints in cls.TOPIC_HINTS.items():
            n = sum(1 for h in hints if fold_vi(h) in low)
            if n > best_n:
                best, best_n = topic, n
        return best

    @staticmethod
    def _sniff_lang(text: str) -> str:
        # Detect Vietnamese on the RAW text, before fold_vi() strips the
        # diacritics -- checking the folded string made has_vi permanently
        # False, so a fully-Vietnamese memory was labelled "en".
        has_vi = any(c in _ACCENTED for c in unicodedata.normalize("NFC", text).lower())
        has_en = bool(re.search(r"\b(the|and|how|what|with|for)\b", text.lower()))
        if has_vi and has_en:
            return "mix"
        return "vi" if has_vi else "en"


if __name__ == "__main__":
    # tiny smoke test
    a = HybridMemoryAgent()
    a.remember("Tài liệu về Kubernetes auto-scaling cho production tiếng Việt")
    a.remember("Hướng dẫn OAuth JWT và zero-trust security")
    print(a.recall("Kubernetes scaling", a.user_id))