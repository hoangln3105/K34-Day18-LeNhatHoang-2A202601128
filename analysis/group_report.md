# Group Report — Lab 18: Production RAG

**Hình thức:** Bài cá nhân (theo ASSIGNMENT.md — implement toàn bộ 5 modules)
**Học viên:** Lê Nhật Hoàng · **Mã:** 2A202601128 · **Lớp:** AICB-K34
**Ngày:** 18/08/2026

## Thành viên & Phân công

Bài cá nhân — một người thực hiện toàn bộ 5 modules.

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Lê Nhật Hoàng | M1: Chunking | ☑ | 13/13 |
| Lê Nhật Hoàng | M2: Hybrid Search | ☑ | 5/5 |
| Lê Nhật Hoàng | M3: Reranking | ☑ | 5/5 |
| Lê Nhật Hoàng | M4: Evaluation | ☑ | 4/4 |
| Lê Nhật Hoàng | M5: Enrichment | ☑ | 10/10 |
| | **Tổng** | | **37/37 (100%)** |

## Kết quả RAGAS

Tập test 20 câu · Corpus 26 documents

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.7583 | **0.7875** | +0.0292 |
| Answer Relevancy | 0.6232 | 0.6176 | −0.0056 |
| Context Precision | 0.9625 | **0.9750** | +0.0125 |
| Context Recall | 0.9250 | 0.9250 | 0.0000 |

⚠️ Chạy baseline 3 lần với code không đổi cho biên độ dao động **±0.069 (faithfulness)** và **±0.083 (answer relevancy)**. Các Δ ở trên đều nhỏ hơn biên độ này nên **nằm trong vùng nhiễu** — chi tiết ở `failure_analysis.md`.

## Key Findings

1. **Biggest improvement:** Sửa chunking để không cắt vỡ bảng markdown, cộng với Small-to-Big (retrieve child → return parent). Faithfulness đi từ 0.6596 → 0.6750 → **0.7875**, đưa production từ chỗ **thua baseline cả 4 metrics** lên vượt baseline.

2. **Biggest challenge:** Production ban đầu tệ hơn baseline. Nguyên nhân chỉ tìm ra khi in chunk thật ra xem: `child_size = 256` cắt đúng vào ô đáp án của bảng thẩm quyền phê duyệt, làm mất `| Trên 50.000.000 VNĐ | Tổng Giám đốc (CEO) |`. Hybrid search + rerank + enrichment đều vô nghĩa khi dữ liệu đã hỏng từ khâu chunking.

3. **Surprise finding:** **3 trong 5 câu tệ nhất thực ra có câu trả lời ĐÚNG hoàn toàn.** Ví dụ câu "Khi phát hiện malware, nhân viên có nên tự xử lý không?" trả lời chính xác nhưng `answer_relevancy` = 0.000 với `context_precision` = `context_recall` = 1.0. RAGAS yếu với tiếng Việt, câu yes/no và multi-hop. Nhìn aggregate mà không đọc output thật sẽ tối ưu nhầm chỗ.

## Presentation Notes (5 phút)

1. **Demo case study (2 phút):** chiếu song song child chunk bị cắt vs basic chunk còn nguyên bảng — cho thấy trực quan vì sao pipeline "xịn" lại thua pipeline naive.
2. **Small-to-Big (1 phút):** search bằng child 256 ký tự để precision cao, trả parent 2048 ký tự để LLM đủ ngữ cảnh. Đây là phần đề bài yêu cầu mà scaffold chưa implement.
3. **Metric artifact (1.5 phút):** trình bày 2 nhóm lỗi — lỗi thước đo (3 câu) vs lỗi retrieval thật (2 câu), nhấn mạnh thông điệp "đọc output trước khi tin metric".
4. **Bàn về variance (0.5 phút):** ±0.08 giữa các lần chạy trên tập 20 câu → cần ≥100 câu và chạy nhiều lần mới kết luận được.

## Ghi chú kỹ thuật

- **Bonus đạt được:** Enrichment combined mode 1 call/chunk (`_enrich_single_call`) — 109 call thay vì 436, tiết kiệm 75% cost · Latency breakdown đầy đủ ở cuối `failure_analysis.md`
- **Hạn chế đã biết:** 2 file PDF scan (`BCTC.pdf`, `Nghi_dinh_13-2023`) bị loại khỏi index vì không có text layer, cần OCR
- **Sửa thêm ngoài 5 module:** `main.py` dùng `os.rename()` gây `FileExistsError` trên Windows ở lần chạy thứ 2 → đổi sang `os.replace()` để pipeline exit code 0
