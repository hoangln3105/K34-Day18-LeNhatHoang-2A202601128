# Failure Analysis — Lab 18: Production RAG

**Học viên:** Lê Nhật Hoàng · **Mã:** 2A202601128 · **Lớp:** AICB-K34
**Bài cá nhân** — implement toàn bộ 5 modules (M1–M5)

---

## RAGAS Scores

Tập test: 20 câu hỏi · Corpus: 26 documents (25 `.md` + 1 PDF có text layer)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7583 | **0.7875** | **+0.0292** |
| Answer Relevancy | 0.6232 | 0.6176 | −0.0056 |
| Context Precision | 0.9625 | **0.9750** | **+0.0125** |
| Context Recall | 0.9250 | 0.9250 | 0.0000 |

**Cấu hình so sánh**

| | Naive Baseline | Production |
|---|---|---|
| Chunking | `chunk_basic` 500 ký tự theo paragraph | `chunk_hierarchical` parent 2048 / child 256 (table-aware) |
| Enrichment | không | M5 combined — 1 API call/chunk |
| Search | Dense-only (bge-m3) | Hybrid: BM25 (underthesea) + Dense + RRF |
| Rerank | không | CrossEncoder `bge-reranker-v2-m3`, top-20 → top-3 |
| Context đưa vào LLM | 3 chunk × ~410 ký tự | 3 parent (Small-to-Big) |

### ⚠️ Cảnh báo về độ tin cậy của Δ

Chạy `naive_baseline.py` **3 lần với code không đổi** cho ra kết quả khác nhau:

| Lần chạy | Faithfulness | Answer Relevancy |
|---|---|---|
| 1 | 0.7146 | 0.5739 |
| 2 | 0.7833 | 0.6567 |
| 3 | 0.7583 | 0.6232 |
| **Biên độ dao động** | **±0.069** | **±0.083** |

Nguyên nhân: LLM sinh câu trả lời và LLM-judge của RAGAS đều non-deterministic, trên tập chỉ 20 câu nên 1 câu đổi điểm = 5% tổng.

**Kết luận:** chỉ Δ vượt ~0.08 mới đáng tin. Δ faithfulness +0.029 và context_precision +0.013 **nằm trong vùng nhiễu** — không được kết luận production tốt hơn chỉ dựa vào 1 lần chạy. Muốn kết luận chắc chắn phải chạy ≥5 lần lấy trung bình, hoặc tăng test set lên ≥100 câu.

---

## Bottom-5 Failures

Sắp theo trung bình 4 metrics tăng dần (lấy từ `reports/ragas_report.json`).

### #1 — Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5–50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra cần xác nhận cấu hình kỹ thuật từ phòng CNTT.
- **Got:** "Giám đốc phòng ban sẽ phê duyệt đơn hàng vì giá trị đơn hàng là 30 triệu VNĐ. Cần có xác nhận của phòng CNTT về cấu hình kỹ thuật của laptop trước khi đề xuất."
- **Worst metric:** `answer_relevancy` = 0.000 (faithfulness 0.5 · precision 1.0 · recall 1.0)
- **Error Tree:** Output sai? → **KHÔNG, output ĐÚNG hoàn toàn cả 2 vế** → Context đúng? → có (precision 1.0, recall 1.0) → Query OK? → có
- **Root cause:** **Lỗi của thước đo, không phải lỗi hệ thống.** `answer_relevancy` hoạt động bằng cách bắt LLM sinh ngược câu hỏi từ answer rồi đo cosine với câu hỏi gốc. Câu này là multi-hop 2 vế ("ai phê duyệt" + "cần gì từ CNTT"); answer gộp cả 2 vế nên câu hỏi sinh ngược không khớp hình dạng câu hỏi gốc → cosine thấp → 0.0.
- **Suggested fix:** Không sửa RAG. Với câu multi-hop nên chấm bằng `answer_correctness` (so với ground_truth) thay vì `answer_relevancy`; hoặc dùng embedding tiếng Việt cho RAGAS thay vì embedding mặc định của OpenAI.

### #2 — Tự xử lý malware

- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Tuyệt đối không tự ý xử lý, phải báo trong 1 giờ qua helpdesk hoặc hotline CNTT.
- **Got:** "Không. Nhân viên tuyệt đối **không tự ý xử lý malware** hoặc tìm cách khắc phục mà không có sự hướng dẫn của đội CNTT."
- **Worst metric:** `answer_relevancy` = 0.000 (faithfulness 0.667 · precision 1.0 · recall 1.0)
- **Error Tree:** Output sai? → **KHÔNG, đúng ý phủ định** → Context đúng? → có (1.0 / 1.0) → Query OK? → có
- **Root cause:** Cùng bản chất #1. Đây là câu **negation dạng yes/no**; answer đúng nhưng rất ngắn ("Không. ..."), thông tin để sinh ngược câu hỏi quá ít → `answer_relevancy` sụp về 0. Đây là điểm mù đã biết của metric này với câu yes/no.
- **Suggested fix:** Tách riêng nhóm câu yes/no khi eval, chấm bằng exact-match trên nhãn CÓ/KHÔNG. Nếu vẫn dùng RAGAS, buộc prompt trả lời đủ câu ("Không, vì …") để metric có đủ ngữ liệu.

### #3 — Senior 9 năm thâm niên (multi-hop 2 tài liệu)

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3) = 18 ngày. Lương Senior (P3-P4): 20–35 triệu/tháng.
- **Got:** "…được nghỉ 18 ngày phép năm (15 ngày cơ bản + 3 ngày cộng thêm từ thâm niên). **Lương không được đề cập trong context.**"
- **Worst metric:** `faithfulness` = 0.333 · **`context_recall` = 0.500** · precision 1.0 · relevancy 0.868
- **Error Tree:** Output sai? → **thiếu một nửa** → Context đúng? → **KHÔNG, recall 0.5 — chỉ lấy được tài liệu nghỉ phép, THIẾU `bang_luong_2024.md`** → dừng ở đây, lỗi ở khâu retrieval
- **Root cause:** **Lỗi thật.** Câu multi-hop cần ghép 2 tài liệu độc lập. Retrieval xếp hạng theo độ tương đồng với TOÀN BỘ câu hỏi, mà vế "nghỉ phép" chiếm ưu thế ngữ nghĩa → cả 3 slot top-3 đều bị tài liệu nghỉ phép chiếm chỗ, bảng lương bị đẩy ra ngoài. `RERANK_TOP_K = 3` làm vấn đề nặng thêm.
- **Suggested fix:** Query decomposition — tách câu hỏi thành 2 truy vấn con ("ngày phép Senior 9 năm" + "lương Senior"), retrieve riêng rồi hợp nhất context. Rẻ hơn: tăng `RERANK_TOP_K` lên 5 và ép đa dạng nguồn (mỗi `source` tối đa 2 chunk trong top-k).

### #4 — Hoàn trả chi phí đào tạo

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Phải cam kết làm ít nhất 1 năm; nghỉ sau 8 tháng là trước hạn → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tức là **25.000.000 VNĐ**."
- **Worst metric:** `faithfulness` = 0.500 · **`context_recall` = 0.500** · precision 1.0 · relevancy 0.789
- **Error Tree:** Output sai? → **kết quả đúng nhưng thiếu căn cứ** → Context đúng? → **KHÔNG, recall 0.5 — thiếu điều khoản "cam kết tối thiểu 1 năm"** → lỗi ở retrieval
- **Root cause:** **Lỗi thật.** Đây là câu numeric cần suy luận 2 bước: (1) tra ngưỡng cam kết 1 năm, (2) so 8 tháng < 12 tháng → 100%. Context chỉ lấy được bảng tỉ lệ hoàn trả, thiếu mệnh đề điều kiện. LLM ra đúng số nhưng bằng suy luận riêng → `faithfulness` bị phạt vì claim không truy vết được về context.
- **Suggested fix:** Chunk theo `chunk_structure_aware` cho tài liệu đào tạo để điều khoản cam kết và bảng tỉ lệ nằm chung một section. Bổ sung HyQA enrichment sinh sẵn câu hỏi "nghỉ trước bao lâu thì phải hoàn trả 100%" để bridge vocabulary gap.

### #5 — Nghỉ không lương 20 ngày

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày cần Giám đốc điều hành (CEO). Lưu ý nghỉ trên 14 ngày phải tự đóng phần bảo hiểm.
- **Got:** "Nghỉ phép không lương 20 ngày cần phê duyệt của **Giám đốc điều hành (CEO)**."
- **Worst metric:** `faithfulness` = 0.000 · relevancy 0.804 · precision 1.0 · recall 1.0
- **Error Tree:** Output sai? → **KHÔNG, đáp án CEO ĐÚNG** → Context đúng? → có (precision 1.0, recall 1.0) → Query OK? → có
- **Root cause:** Answer đúng và context đầy đủ nhưng `faithfulness` = 0. Claim "20 ngày → CEO" đòi hỏi khớp khoảng số (20 nằm trong 16–30) — LLM-judge của RAGAS phải tự suy luận số học để verify, và nó thất bại ở bước này. Đây lại là **giới hạn của thước đo**, cùng họ với #1 và #2.
- **Suggested fix:** Với câu tra bảng khoảng số, thêm vào prompt yêu cầu trích nguyên văn dòng bảng làm căn cứ ("Theo bảng: | 16-30 ngày | CEO |") — vừa giúp người đọc kiểm chứng, vừa cho LLM-judge chuỗi verify tường minh.

---

## Tổng hợp: 2 nhóm nguyên nhân

Phân loại bottom-5 cho thấy chúng **không cùng bản chất**:

| Nhóm | Câu | Dấu hiệu | Bản chất |
|---|---|---|---|
| **A — Lỗi thước đo** | #1, #2, #5 | `context_precision` = 1.0 và `context_recall` = 1.0 nhưng metric answer = 0.0, trong khi answer **đối chiếu tay thấy ĐÚNG** | RAGAS yếu với tiếng Việt, câu yes/no, câu multi-hop và suy luận khoảng số. **Không phải lỗi RAG.** |
| **B — Lỗi thật (retrieval)** | #3, #4 | `context_recall` = 0.5 — thiếu đúng 1 trong 2 tài liệu cần ghép | Multi-hop: một vế câu hỏi lấn át vế kia, top-3 bị một nguồn chiếm hết |

**Ý nghĩa:** trong 5 câu tệ nhất chỉ **2 câu là lỗi hệ thống thật**. Nếu nhìn điểm số mà kết luận "RAG kém" rồi đi tinh chỉnh chunking/rerank thì sẽ tối ưu nhầm chỗ cho 3/5 câu. Đây là lý do bắt buộc phải đọc answer thực tế chứ không chỉ nhìn aggregate.

---

## Case Study: bảng markdown bị cắt vỡ

**Question chọn phân tích:** "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"

Ở lần chạy đầu tiên, production **thua baseline cả 4 metrics** (faithfulness 0.6596 vs 0.7146; answer_relevancy 0.4962 vs 0.5739). Câu này trả lời `"Không tìm thấy."` dù tài liệu `mua_sam.md` có đủ thông tin.

**Error Tree walkthrough:**

1. **Output đúng?** → KHÔNG — trả `"Không tìm thấy."`
2. **Context đúng?** → KHÔNG — in chunk ra thì thấy bị cắt cụt:

   ```
   | Giá trị đơn hàng | Người phê duyệt |
   |-------------------|-----------------|
   | Dưới **5.000.000 VNĐ** | Trưởng phòng (Manager) |
   | Từ **5.000.000 - 50.000.000 VNĐ** | Giám đốc phòng ban (Director) |
   | Trên **50.000.000 VNĐ** |        <-- CẮT ĐÚNG Ô ĐÁP ÁN
   ```

   Baseline `chunk_basic` (500 ký tự theo paragraph) giữ nguyên `| Trên 50.000.000 VNĐ | Tổng Giám đốc (CEO) |` → trả lời được.
3. **Query rewrite OK?** → có, không liên quan
4. **Fix ở bước:** **Chunking (M1)** — child 256 ký tự cắt mù theo số ký tự, phá vỡ bảng markdown

**Đã fix — 2 thay đổi:**

| Fix | Nội dung | Kết quả |
|---|---|---|
| **Table-aware chunking** (`_split_structural_blocks`, `_split_table` trong M1) | Bảng markdown + code block là đơn vị nguyên tử, không bao giờ cắt giữa. Bảng quá lớn thì cắt theo dòng nhưng **lặp lại header + separator** ở mỗi mảnh, để mảnh nào cũng tự giải nghĩa được | `context_precision` 0.9500 → 0.9708 |
| **Small-to-Big / parent-return** (`_PARENT_MAP` trong `pipeline.py`) | Đề M1 yêu cầu "retrieve child → return parent" nhưng scaffold chỉ index child rồi trả luôn child. Sửa: search + rerank trên child 256 ký tự (precision cao), rồi **mở rộng thành parent 2048 ký tự** trước khi đưa vào LLM | faithfulness 0.6750 → 0.7875; production **vượt** baseline |

Trước fix, production đưa vào LLM 3 × 256 ≈ 750 ký tự, baseline đưa 3 × 410 ≈ 1230 ký tự — production có **ít hơn gần một nửa** lượng thông tin. Đó là lý do gốc khiến pipeline "xịn hơn" lại thua pipeline naive.

**Bài học:** thêm hybrid search + reranking + enrichment không cứu được một chunking làm hỏng dữ liệu từ đầu. Lỗi ở khâu ingest lan xuống toàn bộ pipeline phía sau.

**Nếu có thêm 1 giờ, sẽ optimize:**

- **Query decomposition cho câu multi-hop** — sửa được trực tiếp #3 và #4, tức 2/2 lỗi thật trong bottom-5
- **Ép đa dạng nguồn trong top-k** (mỗi `source` tối đa 2 chunk) để vế phụ của câu hỏi không bị đẩy khỏi context
- **Đổi metric cho câu yes/no và multi-hop** sang `answer_correctness` để loại hẳn nhóm lỗi A
- **Chạy eval 5 lần lấy trung bình** — với biên độ dao động ±0.08 hiện tại, một lần chạy không đủ căn cứ kết luận
- **OCR 2 file PDF scan** (`BCTC.pdf`, `Nghi_dinh_13-2023`) đang bị loại hoàn toàn khỏi index

---

## Phụ lục: Latency breakdown

Đo trên CPU, corpus 26 documents → 109 child chunks, 20 câu hỏi.

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| M1 Chunking | 0.0s | thuần Python, không đáng kể |
| **M5 Enrichment** | **343.6s** | **56% tổng thời gian** — 109 chunk × 1 API call `gpt-4o-mini` |
| M2 Indexing (BM25 + Dense) | 21.2s | bge-m3 encode 109 chunk trên CPU |
| M3 Load reranker | 0.0s | nhờ `_MODEL_CACHE` cấp class; lần load đầu ~14s |
| Eval 20 queries (search + rerank + LLM) | ~209s | ~10.5s/câu |
| RAGAS 4 metrics × 20 câu | 44.8s | 80 lần gọi LLM-judge |
| **Tổng production** | **618.4s** | |

**Nhận xét:** enrichment chiếm hơn nửa thời gian nhưng chỉ chạy **một lần lúc index**, không ảnh hưởng latency lúc user hỏi. Chi phí lúc query mới là thứ người dùng cảm nhận: ~10.5s/câu, trong đó cross-encoder rerank 20 cặp (query, doc) trên CPU là phần nặng nhất — chuyển sang GPU hoặc `FlashrankReranker` sẽ giảm mạnh.

Chế độ **combined 1 call/chunk** của M5 (`_enrich_single_call`) gộp summary + questions + context + metadata vào 1 request. Nếu gọi riêng 4 kỹ thuật sẽ là 436 call thay vì 109 → tiết kiệm **75% cost và thời gian**.
