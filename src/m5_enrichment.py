from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, re, sys, json as _json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

ENRICH_MODEL = "gpt-4o-mini"
_CLIENT = None


def _client():
    """Tạo OpenAI client 1 lần rồi tái sử dụng (tránh mở connection pool mỗi chunk)."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        _CLIENT = OpenAI()
    return _CLIENT


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def _extractive_summary(text: str, n_sentences: int = 2) -> str:
    """Fallback không cần API: lấy n câu đầu."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        return text
    return ". ".join(sentences[:n_sentences]).rstrip(".") + "."


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            resp = _client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content":
                        "Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt. "
                        "Bản tóm tắt PHẢI ngắn hơn đoạn gốc. Chỉ trả về phần tóm tắt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            summary = (resp.choices[0].message.content or "").strip()
            # Summary dài hơn bản gốc thì vô nghĩa → quay về extractive
            if summary and len(summary) <= len(text):
                return summary
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap:
    user hỏi "nghỉ phép mấy ngày", tài liệu viết "số ngày phép năm").
    """
    if OPENAI_API_KEY:
        try:
            resp = _client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content":
                        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn "
                        f"có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng, không đánh số."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip().split("\n")
            questions = [q.strip().lstrip("0123456789.-) ") for q in raw if q.strip()]
            if questions:
                return questions[:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Extractive fallback: biến câu khẳng định thành câu hỏi thô
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    Chunk lẻ thường mất chủ ngữ ("Số ngày này tăng thêm 1...") → thêm 1 câu
    định vị giúp embedding mang đủ ngữ cảnh.
    """
    if OPENAI_API_KEY:
        try:
            resp = _client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content":
                        "Viết 1 câu ngắn tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu "
                        "và nói về chủ đề gì. Chỉ trả về đúng 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0,
            )
            context = (resp.choices[0].message.content or "").strip()
            if context:
                return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


_DEFAULT_META = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, category, language.
    Metadata này dùng để filter khi search (VD chỉ lấy category="it").
    """
    if OPENAI_API_KEY:
        try:
            resp = _client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content":
                        'Trích xuất metadata từ đoạn văn. Trả về JSON: '
                        '{"topic": "...", "entities": ["..."], '
                        '"category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = _json.loads(resp.choices[0].message.content)
            if isinstance(data, dict):
                return {**_DEFAULT_META, **data}
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return dict(_DEFAULT_META)


# ─── Combined Single-Call Mode ───────────────────────────

_COMBINED_SYSTEM_PROMPT = """Phân tích đoạn văn và trả về JSON đúng schema sau:
{
  "summary": "tóm tắt 2-3 câu bằng tiếng Việt",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."],
               "category": "policy|hr|it|finance", "language": "vi|en"}
}
Chỉ trả về JSON, không giải thích thêm."""


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ (giảm ~75% cost
    và ~75% latency, vì 4 kỹ thuật đều đọc cùng 1 đoạn văn).
    """
    if OPENAI_API_KEY:
        try:
            resp = _client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": _COMBINED_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = _json.loads(resp.choices[0].message.content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Fallback offline: vẫn trả đủ 4 field để pipeline không đổi hành vi
    return {
        "summary": _extractive_summary(text),
        "questions": generate_hypothesis_questions(text) if not OPENAI_API_KEY else [],
        "context": f"Trích từ {source}." if source else "",
        "metadata": dict(_DEFAULT_META),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
