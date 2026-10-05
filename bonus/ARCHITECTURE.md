# Hybrid Memory Agent — Architecture

Personal AI memory for Vietnamese users. POC: `bonus/agent.py` +
`bonus/demo.py`.

Lab concepts carried over: NB2 RRF fusion, NB4 Feast online features + TTL,
NB5 payload-filtered retrieval, NB7 semantic cache, NB8 point-in-time
correctness.

---

## 1. The problem

The brief asks for one assistant that remembers three things which are
fundamentally different in kind, and the whole design follows from taking that
seriously:

- **Episodic memory** grows constantly, is retrieved by *meaning*, and is
  high-volume. Every document read is a new row.
- **Stable profile** barely changes, is retrieved by *key*, and is tiny.
- **Recent activity** changes every second, is never retrieved by key, and is
  only meaningful as an aggregate.

Putting all three in one store is the obvious move and it is wrong, because a
system optimised for one becomes bad at the other two. A vector store is
excellent at "find me things like this" and terrible at "give me user 42's
reading speed right now". A relational store is the reverse. So: three stores,
one assembly step.

---

## 2. Architecture diagram

```
                       ┌─────────────────────────────────────────┐
   user reads a doc    │           WRITE PATH                    │
   ───────────────────▶│                                         │
                       │  chunk (one memory = one chunk)         │
                       │      │                                  │
                       │      ├──────────────▶ ┌────────────────┐ │
                       │      │                │ EPISODIC       │ │
                       │      │                │ Qdrant         │ │
                       │      │                │ payload:       │ │
                       │      │                │  mem_id,       │ │
                       │      │                │  user_id ◀─────┼─┼── isolation
                       │      │                │  topic, ts     │ │   key (NB5)
                       │      │                └────────────────┘ │
                       │      │                                  │
                       │      ├──────────────▶ ┌────────────────┐ │
                       │      │                │ PROFILE        │ │
                       │      │                │ Feast online   │ │
                       │      │                │ TTL = 30 d     │ │
                       │      │                │  topic_affinity│ │
                       │      │                │  pref_language │ │
                       │      │                │  reading_speed │ │
                       │      │                └────────────────┘ │
                       │      │                     ▲ daily batch   │
                       │      │                     │              │
                       │      ├──────────────▶ ┌────────────────┐ │
                       │      │                │ RECENT         │ │
                       │      │                │ in-process     │ │
                       │      │                │ sliding window │ │
                       │      │                │ TTL = 1 h      │ │
                       │      │                └────────────────┘ │
                       └─────────────────────────────────────────┘

                       ┌─────────────────────────────────────────┐
   "what do I know    │           READ PATH                    │
    about scaling?" ─▶│                                         │
                       │  1. RECORD query into RECENT (1 h TTL)  │
                       │                                         │
                       │  2. CACHE probe  ──── hit  ──▶ done ◀───┼── saves 0 (NN)
                       │     (semantic, namespaced by user)       │
                       │     │ miss                              │
                       │  3. RETRIEVE                           │
                       │     BM25 on diacritic-folded text ─┐   │
                       │     dense cosine on same vectors  ┤   │
                       │     RRF fuse  1/(60 + rank)       ┘   │  (NB2)
                       │     filter to user_id only (NB5)       │
                       │  4. RE-RANK  +0.05 if topic == affinity │
                       │           (soft prior, never a hard    │
                       │            filter — NB5's lesson)      │
                       │  5. ASSEMBLE one context block         │
                       │                                         │
                       │        ┌──────────────────────────┐     │
                       │        │   PROFILE  (who they are)│     │
                       │        │   RECENT   (right now)   │     │
                       │        │   EPISODIC (what they saw)│    │
                       │        └────────────┬─────────────┘     │
                       └─────────────────────┼───────────────────┘
                                             ▼
                                    LLM prompt (vi/en/mix)
```

The asymmetry is the point. Writes fan out to three stores with three
different cadences; reads touch the profile and the recent window by *key*
(O(1), exact) and only the episodic set by *similarity* (approximate). The
expensive approximate search never runs on the cheap data.

---

## 3. Decision 1 — Chunking: one document = one memory

**Chosen:** one saved/finished document becomes exactly one memory chunk,
carrying `mem_id`, `user_id`, `topic`, `created_at` in the payload.

**Rejected:** fixed token-count chunks (~500 tokens with overlap).

**The tradeoff, concretely.** Per-message or fixed-size chunking buys you
finer-grained retrieval: a 40-page PDF becomes 60 vectors, and a question about
one section hits one section. That is a real gain, and it is why production RAG
does it. The cost lands in three places at once. *Storage* — 60× the vectors,
and 60× the re-embedding when the embedding model changes. *Retrieval quality
in the opposite direction* — a chunk is a fragment whose subject is often
ambiguous out of context, so a one-chunk question matches the wrong fragment
three times out of four; you then need a parent-document lookup to recover the
surrounding argument, which is a second system. *Context window* — 60 chunks
returned to the LLM is not 60 facts, it is mostly redundant, and the
per-document memo the assistant shows the user ("here's what you read on X")
becomes impossible to reconstruct.

One-document-per-memory is chosen because the unit of meaning here is the
*artifact the user engaged with*, not the sentence. "What have I read about
Kubernetes?" wants documents. Documents also make the read receipt, the
deduplication, and the forgetting policy (bonus extension: drop untouched
memories after 30 days) operate on a key the user recognises.

The compromise is explicit rather than hidden: memories longer than ~2000
characters are stored whole but *truncated for embedding* at 2000 chars, so the
vector reflects the head of the document where the title and thesis live. The
full text is kept in the payload for the LLM. The vector decides *ranking*, the
payload decides *reading* — they do not need to be the same string.

---

## 4. Decision 2 — Feature schema: tabular profile, vector memory, no embeddings in the feature store

Profile features are all tabular and point-in-time correct:

| feature | entity | ttl | source | why this ttl |
|---|---|---|---|---|
| `topic_affinity` | user | 30 d | daily batch over memory log | a preference measured over weeks should not flicker on one document |
| `preferred_language` | user | 30 d | daily batch | vi/en/mix is stable over months |
| `reading_speed_wpm` | user | 30 d | daily batch | slow-moving trait |
| `queries_last_hour` | user | 1 h | streaming window | a velocity signal; 30 d would make fraud and fatigue detection useless |
| `distinct_topics_24h` | user | 1 h | streaming window | same — this is a *recent spike* detector |

**Chosen:** tabular features for the profile, **rejected:** storing latent
preference as an embedding feature view (a user-embedding vector computed from
their history, looked up by `user_id`).

**The tradeoff.** A user embedding is strictly more expressive than
`topic_affinity` — it captures "this user likes infrastructure *and* dislikes
vendor marketing" in a way no single enum can. That is why I rejected it *for
the profile*, and it costs three things. *Opacity*: a 1024-d vector cannot be
audited, corrected, or shown to a user who asks why they got a recommendation,
and an assistant that misattributes a user's interests cannot be debugged.
*Refresh cost*: a user embedding must be recomputed whenever any new memory
lands — i.e. on every write, not on a schedule. *Coupling*: it puts an ANN
search on the critical path of a lookup that should be an exact primary-key
hit, so the P99 budget for "who is this user" now includes an approximate
index. `topic_affinity` is a string; the P99 target for the NB4 lookup is
under 10 ms and this stays trivially inside it.

The compromise is that latent taste is not thrown away — it is expressed
*downstream*, at re-rank time, as a soft prior (`+0.05` when a memory's topic
matches affinity). A soft prior cannot starve a genuinely off-affinity memory,
which is exactly the failure mode NB5 demonstrates: a hard topic filter on a
narrow facet returns fewer than `top_k` results and quietly loses recall.

Every profile feature is written with a timestamp and read point-in-time. A
recommendation served on Tuesday must not use a preference measured on
Thursday.

---

## 5. Decision 3 — Freshness: three tiers, chosen per use case

| use case | freshness | mechanism | why |
|---|---|---|---|
| "what am I focused on lately?" | **sub-second** | in-process sliding window, appended on each query | the question is about the last few minutes; even 5-minute batch staleness makes the answer wrong |
| "what should I read next?" | **hourly** | topic counts recomputed on recall from the memory log | stable enough that a 1-hour lag is invisible to the user |
| "what do you know about my reading speed?" | **daily** | Feast batch, TTL 30 d | a trait measured over a year does not change by the hour |

**Chosen:** sub-second for recent activity, daily for the profile.
**Rejected:** pushing everything through Feast with a 1-hour TTL, which is the
tempting simplification because it is one system.

**The tradeoff.** One feature store is far less code — no second store to
operate, no second consistency story, one query surface. It also cannot serve
the sub-second case. A 1-hour-TTL feature view is *by construction* up to an
hour stale, so "what am I focused on lately?" would answer with what the user
was doing up to an hour ago. For a fatigue or fraud signal that is not a small
error, it is a non-answer; the user knows what they just typed. The
in-process window is a deliberate exception to "the feature store owns
features": it is a ring buffer over events, not a queryable feature surface,
and it is the one thing here that is allowed to be eventually-correct-but-not-
yet. The cost of the exception is that this one signal is per-process and lost
on restart — acceptable for a POC, and the first thing to fix in production
(durable log, or Feast's Push API).

Note the inversion worth being explicit about: the *fastest* data is the
*least* durable and the *most* durable data is the *slowest*. Freshness is not
a quality ranking.

---

## 6. Rejected alternative: episodic memory as a Feast feature view

I considered unifying the design by storing memories in a Feast
`FeatureView` (entity = `user`, source = Parquet, `memories_30d` as a list or
JSON blob, TTL 30 d) and serving recall from `get_online_features()`. One
store, one registry, one materialisation pipeline, no Qdrant.

I rejected it because the **re-index cycle and the access pattern are
incompatible**, not because it was harder to write.

- Memories arrive continuously — one per document read, unbounded. Profile
  features arrive on a schedule. Putting both in Feast means either
  re-materialising the whole memory set whenever one memory is added, or
  accepting that the online store is a lagging snapshot of the user's history.
  A snapshot is a fine basis for "topic affinity over 30 days" and a terrible
  basis for "what did I just read".
- Recall is a similarity search over the candidate set. Feast's online store
  gives exact key-value and (via a warehouse) batch aggregation. Delivering
  ANN from it means reimplementing HNSW in the retrieval layer — reimplementing
  the one component that is genuinely hard to get right.
- Concretely, it inverts the deletion story: forgetting a memory (a user
  deletes a document) is a one-row Qdrant delete, but a
  `materialize-incremental` that must *not* resurrect it — a different class of
  bug, and the kind that surfaces as a privacy complaint.

So: Feast owns the slow, small, typed, point-in-time data. Qdrant owns the
fast, large, unstructured, similarity-searched data. Neither is asked to do the
other's job.

---

## 7. Vietnamese-context decisions

**Diacritic folding for the lexical half.** Users type `docs ve Kubernetes` as
readily as `tài liệu về Kubernetes`. The BM25 retriever runs on text
normalised to NFC then stripped of diacritics and lowercased, so both forms
produce the same tokens. The cost is admitted: folding collapses distinct
words (`ve`/`về`, `hoa`/`hóa`) and can create collisions. For a personal
memory index of a few thousand documents that trade is worth it; for
open-domain search it would not be, and a real system would add a
syllable-aware matcher (`pyvi`, `underthesea`) on top rather than replace
folding with it.

**Tokenizer choice.** The lab core tokenises on whitespace
(`app/search.py:_tokenize`), which is wrong for Vietnamese — a single word can
be three syllables. Here we use regex word tokens *after* folding, so
`auto-scaling` stays one token and punctuation never becomes a term. The
honest limitation: this is still not a Vietnamese word segmenter. The dense
retriever is what carries Vietnamese paraphrase, and the default
`bge-small-en-v1.5` is English-focused — which is precisely why NB2's
Vietnamese-paraphrase recall is weak on the lite path. The
`EMBEDDING_BACKEND` switch exists for this: `bge-m3` is the correct production
choice for a Vietnamese-first assistant, and it changes the vector dimension,
so the index must be rebuilt.

**Code-switching is a first-class language value, not an error.** Text mixing
Vietnamese and English inside one sentence is normal for this user base, so
`preferred_language` is `vi | en | mix`, and `mix` is a real state rather than a
fallback. It is detected on the *raw* text, before folding strips the diacritics
that are the only evidence Vietnamese was present.

**Privacy — Decree 13 / Nghị định 13.** Personal reading history is sensitive
personal data under Vietnam's personal data protection framework, and
retention must be explicit. Design consequences baked in rather than deferred:
every episodic memory carries a `user_id` payload filter, so isolation is a
query-time property, not an application convention — one missing `where`
clause cannot leak another user's documents. Reading history is never written
into the feature store (a profile feature would replicate it into a second
system with its own backup and access rules), and "what I read" is kept out of
the model context that leaves the machine: the assembled context carries counts
and affinities, not a verbatim transcript. Missing before shipping: encryption
at rest, a per-user delete path that also purges derived `topic_affinity`, and
consent for any model training on the memory log.

---

## 8. What this POC does not handle yet

- **Multi-process durability.** The recent-activity window is per-process. Two
  API replicas each see half the queries, and a restart loses the window.
  Fix: a durable event log feeding Feast's Push API.
- **Encryption and key management.** Nothing is encrypted at rest, and the
  in-memory Qdrant has no per-user key separation. User A's data is isolated by
  filter, not by cryptography.
- **Consolidation.** Nothing collapses five similar memories into one weekly
  summary, so a user with 3,000 documents pays full retrieval cost forever and
  context fills with near-duplicates. A similarity-triggered consolidation job
  is the obvious next step.
- **Cross-device sync.** Memories are local to one process. Two phones have two
  disjoint memories.
- **Deletion semantics.** Deleting a memory does not recompute `topic_affinity`,
  so a profile can keep asserting a preference derived from a document the user
  asked to erase — the exact Decree 13 problem, unhandled.
- **Evaluation.** There is no labelled set of "what should this user recall",
  so the re-rank weight (`+0.05`) is a guess that happens to be safe because
  it is small, not a tuned value.
