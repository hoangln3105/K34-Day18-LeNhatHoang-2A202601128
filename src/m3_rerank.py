from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Cross-encoder rerank: đọc (query, doc) CÙNG LÚC trong 1 forward pass
    → bắt được tương tác từ-với-từ mà bi-encoder (embedding rời) bỏ lỡ.
    Đổi lại: chậm hơn nhiều → chỉ chạy trên top-20 của retrieval, không chạy trên toàn corpus.
    """

    # cache model theo tên: mỗi lần new CrossEncoderReranker() không load lại từ đĩa
    _MODEL_CACHE: dict = {}

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            cached = CrossEncoderReranker._MODEL_CACHE.get(self.model_name)
            if cached is not None:
                self._model = cached
                return self._model
            try:
                # ⚠️ Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
                # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                CrossEncoderReranker._MODEL_CACHE[self.model_name] = self._model
            except Exception as e:
                print(f"  ⚠️  Không load được cross-encoder '{self.model_name}': {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict],
               top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        if model is None:
            # Fallback: giữ nguyên thứ tự retrieval để pipeline không chết
            return [
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i,
                )
                for i, doc in enumerate(documents[:top_k])
            ]

        pairs = [(query, doc.get("text", "")) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)

        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional — dùng khi latency quan trọng hơn accuracy."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from flashrank import Ranker
                self._model = Ranker()
            except Exception as e:
                print(f"  ⚠️  Không load được flashrank: {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict],
               top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        model = self._load_model()
        if model is None:
            return []
        try:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": d.get("text", "")} for i, d in enumerate(documents)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"  ⚠️  Flashrank rerank failed: {e}")
            return []

        out = []
        for i, r in enumerate(results[:top_k]):
            doc = documents[r.get("id", i)]
            out.append(RerankResult(
                text=r.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(r.get("score", 0.0)),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return out


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
    print(benchmark_reranker(reranker, query, docs, n_runs=3))
