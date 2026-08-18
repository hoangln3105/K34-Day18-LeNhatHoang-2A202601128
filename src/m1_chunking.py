from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


_SENTENCE_MODEL = None  # cache model dùng chung cho semantic chunking


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def _get_sentence_model(name: str = "all-MiniLM-L6-v2"):
    """Lazy-load + cache embedding model (tránh load lại mỗi lần gọi)."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer(name)
    return _SENTENCE_MODEL


def split_sentences(text: str) -> list[str]:
    """Tách text thành câu: theo dấu kết câu hoặc dòng trống."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None,
                   min_chunk_chars: int = 100) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Thuật toán:
      1. Tách câu → embed từng câu (all-MiniLM-L6-v2).
      2. Duyệt tuần tự, so cosine(sent[i-1], sent[i]).
      3. sim < threshold → ranh giới chủ đề → mở chunk mới.
      4. min_chunk_chars: chỉ cho phép cắt khi chunk hiện tại đã đủ dài,
         tránh vỡ vụn ở các dòng ngắn (heading, bullet 1 dòng).
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0],
                      metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    embeddings = _get_sentence_model().encode(sentences)

    def cosine(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        current_len = sum(len(s) + 1 for s in groups[-1])
        sim = cosine(embeddings[i - 1], embeddings[i])
        if sim < threshold and current_len >= min_chunk_chars:
            groups.append([sentences[i]])          # ranh giới ngữ nghĩa → chunk mới
        else:
            groups[-1].append(sentences[i])        # cùng chủ đề → gộp tiếp

    return [
        Chunk(text=" ".join(g),
              metadata={**metadata, "chunk_index": i, "strategy": "semantic",
                        "n_sentences": len(g)})
        for i, g in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _is_table_line(line: str) -> bool:
    """Dòng thuộc bảng markdown: bắt đầu (sau khi trim) bằng '|'."""
    return line.strip().startswith("|")


def _split_structural_blocks(text: str) -> list[tuple[str, str]]:
    """Tách text thành các block có nhãn: ("table"|"code"|"text", nội_dung).

    Bảng markdown và code fence được giữ NGUYÊN KHỐI — đây là các cấu trúc mà
    cắt giữa chừng sẽ làm mất nghĩa hoàn toàn (VD cắt mất cột 'Người phê duyệt'
    của dòng cuối bảng → chunk không còn trả lời được câu hỏi nào).
    """
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    kind = "text"

    def flush():
        nonlocal buf, kind
        if buf:
            content = "\n".join(buf).strip()
            if content:
                blocks.append((kind, content))
        buf = []

    in_code = False
    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                buf.append(line)
                flush()
                in_code = False
                kind = "text"
            else:
                flush()
                in_code = True
                kind = "code"
                buf.append(line)
            continue

        if in_code:
            buf.append(line)
            continue

        if _is_table_line(line):
            if kind != "table":
                flush()
                kind = "table"
            buf.append(line)
        else:
            if kind == "table":
                flush()
                kind = "text"
            buf.append(line)

    flush()
    return blocks


def _split_table(table: str, max_size: int) -> list[str]:
    """Cắt bảng quá lớn theo DÒNG, lặp lại header + separator ở mỗi mảnh.

    Nhờ vậy mảnh nào cũng tự giải nghĩa được: đọc '| Trên 50.000.000 VNĐ | CEO |'
    mà không có header thì không biết 2 cột đó là gì.
    """
    lines = [ln for ln in table.split("\n") if ln.strip()]
    if len(lines) <= 2:
        return [table]

    header = lines[:2]                      # dòng tiêu đề + dòng '|---|---|'
    header_text = "\n".join(header)
    body = lines[2:]

    parts: list[str] = []
    current: list[str] = []
    for row in body:
        candidate = header_text + "\n" + "\n".join(current + [row])
        if current and len(candidate) > max_size:
            parts.append(header_text + "\n" + "\n".join(current))
            current = [row]
        else:
            current.append(row)
    if current:
        parts.append(header_text + "\n" + "\n".join(current))
    return parts


def _split_to_size(text: str, max_size: int) -> list[str]:
    """Chia text thành các mảnh <= max_size, TÔN TRỌNG cấu trúc markdown.

    Thứ tự ưu tiên khi cắt:
      1. Không bao giờ cắt giữa bảng / code block (giữ nguyên khối).
         Bảng vượt max_size thì cắt theo dòng + lặp header.
      2. Văn bản thường: cắt ở ranh giới câu.
      3. Câu quá dài: cắt ở khoảng trắng.
    """
    pieces: list[str] = []

    for kind, block in _split_structural_blocks(text):
        if kind == "table":
            # Bảng vừa trong max_size → giữ nguyên khối (kể cả hơi vượt cũng ưu tiên nguyên vẹn)
            if len(block) <= max_size:
                pieces.append(block)
            else:
                pieces.extend(_split_table(block, max_size))
            continue

        if kind == "code":
            pieces.append(block)            # code block luôn giữ nguyên
            continue

        for sentence in split_sentences(block) or [block]:
            while len(sentence) > max_size:
                cut = sentence.rfind(" ", 0, max_size)
                if cut <= 0:
                    cut = max_size
                pieces.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                pieces.append(sentence)

    # Gom các mảnh nhỏ liền kề lại cho đủ dài, nhưng không gộp xuyên qua bảng/code
    out: list[str] = []
    current = ""
    atomic = {p for kind, blk in _split_structural_blocks(text) if kind in ("table", "code")
              for p in ([blk] if len(blk) <= max_size else _split_table(blk, max_size))}
    for piece in pieces:
        if piece in atomic:                 # bảng/code đứng riêng một chunk
            if current.strip():
                out.append(current.strip())
                current = ""
            out.append(piece)
            continue
        if current and len(current) + 1 + len(piece) > max_size:
            out.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current.strip():
        out.append(current.strip())
    return out


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) -> return parent (context).
    Day la default recommendation cho production RAG.

    Returns:
        (parents, children) - moi child co parent_id link den parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    # --- 1. Gom paragraph thanh parent (<= parent_size chars) ---
    parent_texts: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > parent_size:                 # paragraph don qua dai -> cat nho
            if current.strip():
                parent_texts.append(current.strip())
                current = ""
            parent_texts.extend(_split_to_size(para, parent_size))
            continue
        if current and len(current) + 2 + len(para) > parent_size:
            parent_texts.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}".strip()
    if current.strip():
        parent_texts.append(current.strip())

    # --- 2. Moi parent -> nhieu child (<= child_size chars), link bang parent_id ---
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for p_idx, ptext in enumerate(parent_texts):
        pid = f"parent_{p_idx}"
        parents.append(Chunk(
            text=ptext,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "chunk_index": p_idx, "strategy": "hierarchical"},
        ))
        for c_idx, ctext in enumerate(_split_to_size(ptext, child_size)):
            children.append(Chunk(
                text=ctext,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "chunk_index": c_idx, "strategy": "hierarchical"},
                parent_id=pid,
            ))

    return (parents, children)


# --- Strategy 3: Structure-Aware Chunking ---


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers -> chunk theo logical structure.
    Giu nguyen tables, code blocks, lists - khong cat giua chung.
    """
    metadata = metadata or {}
    header_re = re.compile(r"^(#{1,3})\s+(.+)$")
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    breadcrumb: list[str] = []          # duong dan header hien tai, vd ["# A", "## B"]

    def flush(content: str) -> None:
        """Tao 1 chunk = breadcrumb headers + noi dung section (giu nguyen header)."""
        content = content.strip()
        if not content:
            return                      # header lien tiep nhau -> chua co noi dung
        section = " > ".join(h.lstrip("# ").strip() for h in breadcrumb)
        body = "\n\n".join([*breadcrumb, content]).strip()
        chunks.append(Chunk(
            text=body,
            metadata={**metadata, "section": section or "(no section)",
                      "level": len(breadcrumb), "chunk_index": len(chunks),
                      "strategy": "structure"},
        ))

    for part in parts:
        if not part:
            continue
        m = header_re.match(part.strip())
        if m:
            level = len(m.group(1))
            # header moi -> bo cac header cung cap / sau cap khoi breadcrumb
            breadcrumb = [h for h in breadcrumb if len(h.split(" ")[0]) < level]
            breadcrumb.append(part.strip())
        else:
            flush(part)

    # Toan bo doc chi co header, khong co body -> van giu title lai
    if not chunks and breadcrumb:
        section = " > ".join(h.lstrip("# ").strip() for h in breadcrumb)
        chunks.append(Chunk(text="\n\n".join(breadcrumb),
                            metadata={**metadata, "section": section, "level": len(breadcrumb),
                                      "chunk_index": 0, "strategy": "structure"}))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
