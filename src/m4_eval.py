from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value) -> float:
    """NaN / None / non-numeric → 0.0 (RAGAS trả NaN khi metric không tính được)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v          # v != v ⇔ NaN


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation — 4 metrics.

    - faithfulness      : answer có bịa so với context không (hallucination)
    - answer_relevancy  : answer có trả lời đúng câu hỏi không
    - context_precision : context lấy về có bao nhiêu phần thực sự liên quan (nhiễu)
    - context_recall    : context có đủ thông tin so với ground_truth không

    Bọc trong try/except: RAGAS cần OPENAI_API_KEY + Python 3.11+ (asyncio).
    Thiếu key → trả 0.0 để pipeline vẫn chạy end-to-end.
    """
    empty = {m: 0.0 for m in METRIC_NAMES} | {"per_question": []}

    if not questions:
        return empty

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_safe_float(row.get("faithfulness")),
                answer_relevancy=_safe_float(row.get("answer_relevancy")),
                context_precision=_safe_float(row.get("context_precision")),
                context_recall=_safe_float(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]

        # Aggregate = trung bình per-question (NaN đã quy về 0.0)
        n = max(len(per_question), 1)
        aggregate = {
            m: sum(getattr(r, m) for r in per_question) / n
            for m in METRIC_NAMES
        }
        return {**aggregate, "per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return empty


# ─── Diagnostic Tree: metric thấp nhất → nguyên nhân → cách sửa ─────────

DIAGNOSTIC_TREE: dict[str, tuple[str, str]] = {
    "faithfulness": (
        "LLM hallucinating — câu trả lời chứa thông tin không có trong context",
        "Siết prompt ('CHỈ dùng context, không suy đoán'), hạ temperature về 0, "
        "bắt LLM trích dẫn nguyên văn trước khi kết luận",
    ),
    "context_recall": (
        "Missing relevant chunks — retrieval bỏ sót thông tin cần thiết",
        "Tăng top_k, cải thiện chunking (hierarchical/structure-aware), "
        "thêm BM25 vào hybrid để bắt keyword hiếm, thêm HyQA enrichment",
    ),
    "context_precision": (
        "Too many irrelevant chunks — context lẫn nhiều nhiễu, chunk đúng bị đẩy xuống dưới",
        "Bật/ tăng cường reranking (cross-encoder), giảm RERANK_TOP_K, "
        "thêm metadata filter (version/category) để loại tài liệu hết hiệu lực",
    ),
    "answer_relevancy": (
        "Answer doesn't match question — trả lời lạc đề hoặc quá chung chung",
        "Sửa prompt template (yêu cầu trả lời trực tiếp, đúng đơn vị/con số), "
        "thêm few-shot example, xử lý riêng câu hỏi multi-hop / phủ định",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Với mỗi câu: lấy trung bình 4 metric → sắp xếp tăng dần → lấy bottom-N.
    Metric thấp nhất của câu đó quyết định diagnosis + suggested_fix.
    """
    if not eval_results:
        return []

    scored = []
    for r in eval_results:
        metrics = {m: getattr(r, m) for m in METRIC_NAMES}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "avg_score": round(avg, 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "n_contexts": len(r.contexts),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda d: d["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
