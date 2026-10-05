# Reflection — Lab 19

**Tên:** Nguyễn Hải Đăng - 2A202602963
**Cohort:** 4
**Path đã chạy:** lite (không Docker) + GPU NVIDIA cho embedding

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**Đo được** (Precision@10, NB2 §4–5):

| loại query | BM25 | vector | hybrid |
|---|---|---|---|
| `exact` (15) | **96.7%** | 88.7% | 96.7% |
| `paraphrase` (15) | **33.3%** | 24.0% | 32.0% |
| `mixed` (20) | 97.0% | 98.5% | **100%** |

BM25 thắng `exact`: query có đúng token kỹ thuật, `idf` chấm từ hiếm cao.
Hybrid thắng `mixed` (100%) — truy vấn thật có cả từ khóa verbatim lẫn diễn đạt
lại ý; `1/(60+rank)` nên một retriever có tín hiệu là đủ để doc leo lên.

Vector **không** thắng ở slice nào, và nguyên nhân không phải thuật toán:
`bge-small-en-v1.5` train trên *tiếng Anh*, lab là tiếng Việt. Giữ mọi thứ, chỉ
đổi model: `paraphrase` nhảy **24.0% → 74.7%** (NB2 §6).

**Không dùng hybrid khi:** (1) truy vấn có cấu trúc cố định (SKU, mã CI, stack
trace) — BM25 thắng tuyệt đối, P50 hybrid gấp ~4× BM25; (2) vector index chưa
re-embed xong — nhánh vector trả kết quả cũ, làm nhiễu fusion; (3) thiếu compute
cho query embedding; (4) cần giải thích kết quả.

Hybrid chỉ **+0.8pp** so với BM25: nhỏ *vì lý do sửa được* — đúng về *kiến trúc*,
nhưng không cứu được model sai.

---

## Điều ngạc nhiên nhất khi làm lab này

Đo được `paraphrase` chỉ 24% — *thấp hơn cả BM25* — trong khi vector search tồn
tại để giải quyết đúng việc đó. Phản ứng mặc định là đổ RRF `k`, nhưng đó không
phải công thức sai: `bge-small-en-v1.5` train trên corpus *tiếng Anh*, còn lab là
tiếng Việt. Giữ nguyên BM25 + RRF k=60 + golden set, chỉ đổi model sang
`multilingual-e5-large`, recall nhảy 24.0% → 74.7%.

Bài học: metric tệ không phải lúc nào cũng là lỗi thuật toán, đôi khi là lỗi chọn
model — và cách phân biệt là thay từng biến một, giữ nguyên phần còn lại.

---

## Bonus challenge

- [x] Đã làm bonus — xem [`../bonus/`](../bonus/)
  - `bonus/ARCHITECTURE.md` — diagram + 3 quyết định kiến trúc (chunking,
    feature schema, freshness) + 1 phương án bị loại có lý do + góc nhìn Việt
    Nam (code-switching, diacritic folding, Nghị định 13).
  - `bonus/agent.py` — `HybridMemoryAgent.remember()` + `.recall()`.
  - `bonus/demo.py` — 5 query + cache round-trip + kiểm tra isolation đa user,
    exit 0.
- [ ] Pair work với: _(không làm pair)_

Quyết định đáng chú ý nhất: **loại bỏ** ý tưởng lưu episodic memory thành một
Feast feature view, dù sẽ gộp cả lab về một hệ thống. Lý do không phải độ khó
code mà là **chu kỳ re-index không tương thích** — profile feature đến theo lịch,
còn memory đến liên tục mỗi lần user đọc tài liệu. Nếu để chung, hoặc phải
materialize lại toàn bộ mỗi lần thêm một memory, hoặc chấp nhận online store chỉ
là snapshot trễ — vô dụng với câu hỏi "tôi vừa đọc gì".

---

## Ghi chú kỹ thuật

**GPU cho embedding** (`bash setup-gpu.sh`, xem `make device`): embedding 1000
docs 36 s (CPU) → 1.2 s (GPU); hybrid query P50 8.0 ms → 3.9 ms;
`multilingual-e5-large` index 235 s → 13 s. **Precision@10 giống hệt trên cả
hai** — GPU chỉ đổi tốc độ, không đổi kết quả, và đó là điều duy nhất đáng tin để
khẳng định tối ưu hợp lệ.

Cái bẫy đáng nhớ: `pip install onnxruntime-gpu` **không** kéo theo CUDA runtime, và
khi provider load thất bại, onnxruntime in một dòng ERROR rồi **rơi về CPU và vẫn
trả về vector đúng** — hệ thống chạy CPU mà không có dấu hiệu gì sai. Nên
`make device` không hỏi "provider nào khả dụng" mà kiểm tra chính session
onnxruntime xem nó thực sự bind provider nào.

**Hai lỗi tìm được khi kiểm tra lại lab:**

1. **`.env` chưa từng được đọc.** `setup-docker.sh` sửa `.env` thành
   `QDRANT_MODE=server` + `EMBEDDING_BACKEND=bge-m3`, nhưng không module nào gọi
   `getenv` cho các biến đó. Docker path vẫn chạy và vẫn in kết quả hợp lệ, chỉ là
   đang đo sai hệ thống: Qdrant in-memory thay vì server, vector 384d thay vì
   1024d. Loại bug tệ nhất có thể gặp — không nổ, không đỏ, chỉ sai.

2. **`bonus/agent.py` không import được.** Bảng fold dấu tiếng Việt lệch độ dài
   74 vs 69 nên `ValueError` ngay dòng đầu, cộng 4 lỗi API khác; một
   `except Exception` đã che mất phần lớn. Bonus 7/20 điểm của `agent.py` +
   `demo.py` chỉ cần chạy được là 0 nếu không sửa.