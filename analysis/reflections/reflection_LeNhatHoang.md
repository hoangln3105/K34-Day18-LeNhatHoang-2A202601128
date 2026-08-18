# Individual Reflection — Lab 18: Production RAG

**Tên:** Lê Nhật Hoàng · **Mã học viên:** 2A202601128 · **Lớp:** AICB-K34
**Module phụ trách:** toàn bộ M1 → M5 (bài cá nhân)
**Số tests pass:** 37/37 (M1 13/13 · M2 5/5 · M3 5/5 · M4 4/4 · M5 10/10)

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật đo được) |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 trên corpus 26 doc tạo **125 chunks** (avg 166 ký tự) so với **51 chunks** của basic (avg 410). Semantic cắt vụn hơn hẳn vì `all-MiniLM-L6-v2` là model tiếng Anh, cosine giữa 2 câu tiếng Việt liên quan vẫn thường < 0.85 → hầu như câu nào cũng thành ranh giới. Phải thêm tham số `min_chunk_chars=100` chặn cắt khi chunk chưa đủ dài, nếu không mỗi heading 1 dòng thành 1 chunk riêng. **Bài học: threshold semantic không mang sang ngôn ngữ khác được.** |
| Hierarchical / Small-to-Big | M1 | `chunk_hierarchical()` + `_PARENT_MAP` trong `pipeline.py` | 26 doc → **11 parents / 106 children** (child avg 197, max đúng 256). Quan trọng: chỉ chunk parent-child thôi **chưa đủ** — scaffold index child rồi trả luôn child, khiến LLM chỉ nhận 750 ký tự context (baseline có 1230). Phải tự implement bước "retrieve child → return parent" thì faithfulness mới từ 0.6750 lên **0.7875**. |
| Structure-aware chunking | M1 | `chunk_structure_aware()`, `_split_structural_blocks()`, `_split_table()` | 106 chunks, avg 221 nhưng **max 825** — chính là các bảng markdown được giữ nguyên khối thay vì cắt ở 256. Đây là fix quan trọng nhất của cả lab (xem Phần 2). |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()`, `segment_vietnamese()` | RRF giải quyết việc **không thể so sánh trực tiếp điểm BM25 (không giới hạn) với cosine (0..1)** — chỉ dùng thứ hạng: `score(d) = Σ 1/(k + rank + 1)`. Test thực tế: doc xuất hiện ở cả 2 list được đẩy lên hạng 1 (0.03252) cao gấp đôi doc chỉ có ở 1 list (0.01639). |
| Vietnamese word segmentation | M2 | `segment_vietnamese()` | `underthesea` trả `"Nhân_viên được nghỉ_phép năm"`. Nếu không `replace("_", " ")` thì BM25 coi `nghỉ_phép` là 1 token, còn query "nghỉ phép" là 2 token → **không bao giờ khớp**. Một dòng code quyết định BM25 chạy hay chết. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` đọc (query, doc) cùng lúc trong 1 forward pass nên bắt được tương tác từ-với-từ mà bi-encoder bỏ lỡ. Latency ~10.5s/câu trên CPU cho 20 cặp — quá chậm cho production, phải dùng GPU hoặc flashrank. `context_precision` đạt **0.9750**, cao nhất trong 4 metrics. |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | Gộp 4 kỹ thuật vào 1 API call: 109 chunk = **109 call thay vì 436** → tiết kiệm 75% cost. Chi phí thật: 343.6s cho 109 chunk (56% tổng thời gian pipeline) nhưng chỉ chạy 1 lần lúc index, không ảnh hưởng latency lúc user hỏi. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Metric thấp nhất là **`answer_relevancy` 0.6176**. Nhưng khi đọc answer thực tế thì **3/5 câu tệ nhất có câu trả lời ĐÚNG hoàn toàn** mà vẫn bị chấm 0.0 — metric hỏng chứ không phải RAG hỏng (chi tiết ở `failure_analysis.md`). |
| Diagnostic / Error Tree | M4 | `DIAGNOSTIC_TREE` + `failure_analysis()` | Map metric thấp nhất → nguyên nhân → cách sửa. Giá trị thật của Error Tree là **phân biệt lỗi retrieval với lỗi generation**: câu có `context_recall = 0.5` là lỗi thật ở retrieval, câu có precision/recall = 1.0 mà answer metric = 0 là lỗi thước đo. |

---

## Phần 2: Khó khăn & cách giải quyết

### Khó khăn 1 — Production TỆ HƠN baseline cả 4 metrics (khó nhất)

**Triệu chứng:** chạy xong lần đầu, pipeline "xịn" (hybrid + rerank + enrichment) lại thua pipeline naive:

```
faithfulness      0.7146 -> 0.6596  (-0.0549)
answer_relevancy  0.5739 -> 0.4962  (-0.0777)
context_recall    0.9250 -> 0.8417  (-0.0833)
```

**Cách debug:** không đoán, mà mở `reports/ragas_report.json` đọc answer thực tế. Phát hiện nhiều câu trả về đúng chuỗi:

```
'Không tìm thấy.'
```

trong khi `context_precision = 1.0` và `context_recall = 1.0` — tức RAGAS nói context có đủ thông tin nhưng LLM vẫn nói không tìm thấy. Mâu thuẫn này chỉ ra lỗi nằm ở **nội dung context**, nên tôi in trực tiếp chunk ra xem:

```python
parents, children = chunk_hierarchical(text, metadata={"source":"mua_sam.md"})
for c in children:
    if "50.000.000" in c.text: print(c.text)
```

Kết quả lộ ra ngay:

```
| Trên **50.000.000 VNĐ** |          <-- cắt đúng ô đáp án, mất "Tổng Giám đốc (CEO)"
```

**Nguyên nhân:** `child_size = 256` cắt mù theo số ký tự, phá vỡ bảng markdown. Baseline dùng chunk 500 ký tự theo paragraph nên bảng còn nguyên → trả lời được. **Chunking hỏng làm sập cả pipeline phía sau, không kỹ thuật nào ở M2/M3/M5 cứu được.**

**Cách giải quyết — 2 fix:**

1. **Table-aware chunking** (`_split_structural_blocks`, `_split_table`): bảng markdown và code block thành đơn vị nguyên tử, không bao giờ cắt giữa. Bảng quá lớn thì cắt theo dòng nhưng lặp lại header + separator ở mỗi mảnh — vì đọc `| Trên 50.000.000 VNĐ | CEO |` mà không có header thì không biết 2 cột đó là gì.
2. **Small-to-Big** (`_PARENT_MAP` trong `pipeline.py`): đề bài M1 ghi rõ "retrieve child → return parent" nhưng scaffold không hề làm. Sửa để search/rerank trên child 256 ký tự (precision cao) rồi mở rộng thành parent 2048 ký tự trước khi đưa vào LLM.

**Kết quả:** faithfulness 0.6596 → 0.6750 (sau fix bảng) → **0.7875** (sau parent-return). Production vượt baseline.

**Chi tiết cần nhớ:** key của `_PARENT_MAP` phải là `f"{source}::{parent_id}"` chứ không phải `parent_id` — vì `parent_id` chỉ unique trong 1 document, cả 26 doc đều có `parent_0`, nếu quên `source` sẽ tra nhầm parent của document khác.

### Khó khăn 2 — Không cài được dependencies (mạng 1.6 KB/s)

**Exact error:**

```
× Failed to download `transformers==5.15.0`
├─▶ Request failed after 6 retries in 129.2s
╰─▶ operation timed out
```

**Cách debug:** đo băng thông thật bằng `curl -w "%{speed_download}"` tới nhiều host để phân biệt "PyPI chậm" với "mạng chậm":

| Host | Tốc độ |
|---|---|
| files.pythonhosted.org | 17 KB/s → tụt còn 1.6 KB/s |
| pypi.tuna.tsinghua.edu.cn | 13 KB/s |
| github.com | 0 B/s (timeout) |

→ Cả mạng chậm, đổi mirror vô ích. Với 1.6 KB/s thì riêng `torch` 230 MB mất ~4 tiếng.

**Cách giải quyết:** tìm tài nguyên đã có sẵn trên máy thay vì tải mới.

- `~/.cache/huggingface/hub` đã có sẵn `bge-m3` (4.35 GB) và `bge-reranker-v2-m3` (2.19 GB) → không cần tải model
- Quét ổ đĩa tìm `site-packages/transformers` → phát hiện venv lab cũ `K4-Day08-RAG-Pipeline` dùng **Python 3.11.5 trùng khớp chính xác** với venv lab này, có sẵn torch 2.13.0, transformers 4.46.3, ragas 0.1.21, datasets, pyarrow
- `robocopy` 1.5 GB site-packages sang venv mới → xong trong vài phút, không tốn mạng
- Phần còn thiếu (qdrant-client, underthesea, pypdf, flashrank) lấy từ uv cache bằng `uv pip install --offline`

**Kiến thức thiếu → cách bổ sung:** trước đây tôi mặc định "cài package = phải có mạng". Giờ hiểu rằng wheel `py3-none-any` dùng lại được giữa các project, còn wheel C-extension thì phải trùng phiên bản Python (ABI) — đó là lý do venv Python 3.11.5 copy được sang 3.11.5 nhưng không copy sang 3.14 được.

### Khó khăn 3 — HuggingFace reset kết nối dù model đã cache

**Exact error:**

```
'(ProtocolError('Connection aborted.', ConnectionResetError(10054,
'An existing connection was forcibly closed by the remote host')))'
thrown while requesting HEAD https://huggingface.co/BAAI/bge-m3/resolve/main/modules.json
Retrying in 1s [Retry 1/5].
```

**Nguyên nhân:** `sentence-transformers` luôn gửi HEAD request kiểm tra model mới trước khi dùng cache, mỗi lần load model mất thêm hàng chục giây retry vô ích.

**Giải quyết:** đặt `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1` → đọc thẳng từ cache, bỏ qua HEAD check.

### Khó khăn 4 — Pipeline exit code 1 dù chạy xong

**Exact error:**

```
FileExistsError: [WinError 183] Cannot create a file when that file already exists:
'ragas_report.json' -> 'reports/ragas_report.json'
```

**Nguyên nhân:** `main.py` dùng `os.rename()`, mà trên Windows hàm này ném lỗi nếu file đích đã tồn tại (khác Linux — ghi đè im lặng). Lần chạy thứ 2 trở đi luôn fail.

**Giải quyết:** đổi sang `os.replace()` — ghi đè trên mọi OS. Quan trọng vì rubric mục #6 chấm đúng `exit code 0`.

### Khó khăn 5 — Test M3 chạy rất lâu

**Nguyên nhân:** mỗi test tạo `CrossEncoderReranker()` mới → load lại model 2.2 GB từ đĩa 5 lần.

**Giải quyết:** thêm `_MODEL_CACHE` cấp class (dict theo tên model), instance mới dùng lại model đã load. 5 tests từ rất lâu xuống **16 giây**.

---

## Phần 3: Action Plan cho project cá nhân

### Project: Hệ thống Q&A nội bộ trên tài liệu chính sách công ty (tiếng Việt)

#### Hiện tại

- RAG pipeline hiện tại: chunking cố định theo số ký tự + dense-only search, chưa có rerank, chưa có eval tự động
- Known issues:
  - Tài liệu chính sách đầy **bảng biểu** (hạn mức, thẩm quyền phê duyệt, khung lương) — đúng loại dữ liệu mà lab này chứng minh là bị chunking cố định phá hỏng
  - Có nhiều **phiên bản tài liệu** (v2023 vs v2024) nhưng chưa lọc theo version → trả lời bằng chính sách hết hiệu lực
  - Không biết hệ thống đúng bao nhiêu %, chỉ test cảm tính

#### Plan áp dụng

1. **Chunking strategy:** `chunk_structure_aware` + table-aware làm mặc định, kết hợp hierarchical parent-child.
   *Tại sao:* lab này chứng minh trực tiếp — bảng bị cắt làm mất đáp án khiến LLM trả "Không tìm thấy" dù context_recall = 1.0. Với tài liệu chính sách nhiều bảng, đây là rủi ro số một.
2. **Search:** Hybrid BM25 + Dense + RRF, BM25 bắt buộc qua `underthesea` và nhớ `replace("_", " ")`.
   *Tại sao:* câu hỏi nội bộ chứa nhiều mã/thuật ngữ chính xác ("PVI", "P3-P4", "MFA") mà dense embedding hay bỏ sót; BM25 bắt keyword hiếm rất tốt. RRF hợp nhất mà không cần normalize điểm.
3. **Reranking:** Có — `bge-reranker-v2-m3`, nhưng **chạy trên GPU**.
   *Tại sao:* CPU cho 10.5s/câu là không dùng được. Nếu không có GPU thì dùng `FlashrankReranker` (<5ms) và chấp nhận giảm chút accuracy.
4. **Evaluation:** RAGAS cho `context_precision` / `context_recall`, nhưng **KHÔNG** dùng `answer_relevancy` cho câu yes/no và multi-hop.
   *Tại sao:* lab này cho thấy 3/5 câu tệ nhất bị chấm 0.0 dù trả lời đúng. Thay bằng `answer_correctness` so với ground_truth, và tự viết test set ≥100 câu (20 câu dao động tới ±0.08 giữa các lần chạy — quá nhiễu để ra quyết định).
5. **Enrichment:** `contextual_prepend` + auto metadata theo chế độ combined 1 call/chunk.
   *Tại sao:* metadata tự động (đặc biệt **version** và **category**) cho phép lọc tài liệu hết hiệu lực — đúng vấn đề v2023/v2024 đang gặp. Combined mode tiết kiệm 75% cost so với gọi riêng 4 kỹ thuật.

#### Timeline

| Tuần | Công việc | Tiêu chí hoàn thành |
|---|---|---|
| **Tuần 1** | Viết test set 100 câu có ground_truth, phủ đủ 6 dạng (lookup, version, negation, multi-hop, numeric, ambiguous). Chạy RAGAS lấy baseline hiện tại | Có số baseline để so, chạy 5 lần lấy trung bình |
| **Tuần 2** | Thay chunking sang structure-aware + table-aware + parent-return. Đo lại | `context_recall` tăng, không còn câu "Không tìm thấy" khi context đủ |
| **Tuần 3** | Thêm BM25 tiếng Việt + RRF. Thêm rerank trên GPU | `context_precision` ≥ 0.90, latency < 2s/câu |
| **Tuần 4** | Enrichment + metadata filter theo version. Xử lý câu multi-hop bằng query decomposition | Không còn trả lời bằng chính sách hết hiệu lực; câu multi-hop lấy đủ 2 nguồn |
| **Tuần 5** | OCR tài liệu scan (lab này bỏ mất 2 PDF vì không có text layer). Đưa eval vào CI | Mọi PR đều chạy RAGAS tự động, có cảnh báo khi điểm tụt |

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | 5 | Map được đủ 5 module vào code, và hiểu **tại sao** chứ không chỉ làm theo |
| Code quality | 4 | 37/37 test pass, 0 TODO, có fallback khi thiếu API key. Trừ 1 vì `chunk_semantic` còn phụ thuộc model tiếng Anh |
| Debugging | 5 | Tự tìm ra lỗi bảng markdown bằng cách đọc dữ liệu thật thay vì đoán; xử lý được cả sự cố môi trường không liên quan tới bài |
| Problem solving | 5 | Vượt qua mạng 1.6 KB/s bằng cách tái dùng venv cũ + HF cache thay vì ngồi chờ tải |

### Điều bất ngờ nhất

**Điểm số cao chưa chắc là hệ thống tốt, điểm số thấp chưa chắc là hệ thống tệ.** Câu "Khi phát hiện malware, nhân viên có nên tự xử lý không?" được trả lời hoàn toàn chính xác nhưng `answer_relevancy` = 0.0. Nếu chỉ nhìn con số rồi lao vào sửa prompt thì đã tối ưu nhầm chỗ. Bài học lớn nhất của buổi hôm nay không phải là kỹ thuật RAG nào, mà là **luôn đọc output thật trước khi tin vào metric**.
