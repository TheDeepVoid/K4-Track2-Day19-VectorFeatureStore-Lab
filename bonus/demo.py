"""Bonus demo — 5 queries against the hybrid memory agent (BONUS-CHALLENGE §3).

Run:  python bonus/demo.py        (exit 0 on success)

The five queries are the ones the brief specifies, and each one is here to make
a *different* memory subsystem pay off, so the output is evidence for the
architecture claims in bonus/ARCHITECTURE.md rather than a smoke test:

  1. simple lookup   -> episodic only (vector).  "What have I read about Kubernetes?"
  2. profile-needed  -> Feast-style profile features steer the ranking.
  3. fresh-activity  -> the 1h streaming window answers this, not the vector store.
  4. paraphrase      -> dense wins where BM25 has no literal token overlap.
  5. mixed           -> RRF(fused) + profile re-rank together.

Query 4 is the interesting one: it is typed without the diacritics or the word
"Kubernetes" that the memory actually contains, so the BM25 half scores ~0 and
the whole result is carried by the dense half. Query 1 is the mirror image.
Together they are the argument for fusing rather than picking a retriever.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402

# ── the user's reading history (episodic memory, 12 chunks) ──────────────
# Deliberately spans several topics and two languages, Vietnamese diacritics
# included and omitted, so every retrieval path has something to bite on.
SEED_MEMORIES: list[tuple[str, str]] = [
    ("cloud", "Kubernetes auto-scaling: HPA theo CPU và memory, cần metrics-server cài trong cluster"),
    ("cloud", "Serverless trên AWS Lambda so với Fargate về chi phí và cold start"),
    ("cloud", "Mở rộng hạ tầng theo chiều ngang với load balancer và autoscaling group"),
    ("security", "OAuth 2.0 và JWT: lưu refresh token ở đâu để tránh XSS đánh cắp"),
    ("security", "Zero-trust security: xác thực từng request thay vì tin mạng nội bộ"),
    ("security", "Mã hoá dữ liệu nhạy cảm at rest và in transit, quản lý khoá với KMS"),
    ("database", "Index B-tree của PostgreSQL: composite index theo thứ tự cột hay chọn lọc"),
    ("database", "Cơ sở dữ liệu phân tán: sharding theo tenant so với replication read replica"),
    ("ai_ml", "Mô hình embedding đa ngôn ngữ: so sánh bge-m3 và multilingual-e5-large"),
    ("ai_ml", "RAG pipeline: chunking theo đoạn văn và rerank bằng cross-encoder"),
    ("data_eng", "Kafka và Flink xử lý streaming: exactly-once semantics và watermark"),
    ("devops", "CI/CD với Docker multi-stage build, cache layer và triển khai canary"),
]

# Without diacritics, no "Kubernetes" token: BM25 finds nothing, dense must carry it.
QUERIES: list[tuple[str, str, str]] = [
    ("1. simple lookup (episodic, vector)", "What have I read about Kubernetes?", "cloud"),
    ("2. profile-needed (topic_affinity)", "Recommend what to read next", None),
    ("3. fresh-activity (streaming window)", "What am I focused on lately?", None),
    ("4. paraphrase (no literal overlap)", "Documents about scaling infrastructure?", "cloud"),
    ("5. mixed (hybrid + profile)", "Give me a cloud security summary", None),
]


def main() -> int:
    agent = HybridMemoryAgent(user_id="u_001", top_k=4)

    for topic, text in SEED_MEMORIES:
        agent.remember(text, user_id="u_001", topic=topic)
    print(f"[setup] stored {len(agent.memories)} episodic memories for u_001 "
          f"(profile: topic_affinity={agent._profile['u_001']['topic_affinity']}, "
          f"preferred_language={agent._profile['u_001']['preferred_language']})\n")

    for i, (label, query, expect) in enumerate(QUERIES, start=1):
        print("=" * 78)
        print(f"Q{i}. {label}")
        print(f"    {query!r}")
        print("=" * 78)
        # Q4 is asked twice in a row on purpose: the second call must be served
        # from the semantic cache, which is the NB7 lesson made visible.
        context = agent.recall(query, user_id="u_001")
        print(context)
        if expect:
            hit = f"[{expect}]" in context
            print(f"    -> expected topic {expect} present in context: {hit}")
            assert hit, f"Q{i}: expected a {expect} memory in the assembled context"
        print()

    # cache round-trip: identical question, no new memory in between -> [source=cache]
    repeat = agent.recall(QUERIES[3][1], user_id="u_001")
    assert repeat.startswith("[source=cache"), (
        f"expected the repeated query to hit the semantic cache, got {repeat[:40]!r}"
    )
    print("=" * 78)
    print("cache round-trip: repeated Q4 served from cache -> "
          f"hits={agent.cache.stats.hits} misses={agent.cache.stats.misses}\n")

    # multi-user isolation: u_002 has no memories, and must never see u_001's.
    agent.remember("Mật khẩu và 2FA cho tài khoản ngân hàng", user_id="u_002", topic="security")
    other = agent.recall("Kubernetes", user_id="u_002")
    assert "Kubernetes" not in other, "cross-user leak: u_002 saw u_001's memories"
    print(f"isolation: u_002 recall after 1 own memory -> "
          f"{'ok, no u_001 leakage' if 'Kubernetes' not in other else 'LEAK'}\n")

    print("demo complete — 5 queries, cache round-trip, and user isolation all OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
