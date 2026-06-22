from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "ragas_data"
OUT = ROOT / "eval" / "ragas_sample_dataset.csv"

STOPWORDS = {
    "about", "after", "also", "among", "approach", "based", "because", "been",
    "being", "between", "both", "cannot", "could", "data", "does", "during",
    "each", "from", "have", "into", "more", "most", "paper", "proposed",
    "show", "shown", "shows", "such", "than", "that", "their", "these",
    "this", "those", "through", "using", "were", "when", "where", "which",
    "while", "with", "within", "without", "would", "model", "models",
    "method", "methods", "learning", "network", "networks", "training",
}


def question_count(page_count: int) -> int:
    if page_count > 50:
        return 10
    if page_count > 20:
        return 5
    return 3


def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts]


def keyword_phrase(sentence: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", sentence.lower())
    words = [w for w in words if w not in STOPWORDS and not w.isdigit()]
    if not words:
        return "the main finding"
    counts = Counter(words)
    ranked = [w for w, _ in counts.most_common(4)]
    return " ".join(ranked[:3])


def sentence_score(sentence: str) -> float:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", sentence.lower())
    useful = [w for w in words if w not in STOPWORDS]
    has_signal = any(
        term in sentence.lower()
        for term in (
            "propose", "present", "demonstrate", "result", "achieve",
            "outperform", "improve", "introduce", "show", "find", "evaluate",
            "algorithm", "framework", "architecture", "objective", "loss",
        )
    )
    return len(set(useful)) + (6 if has_signal else 0) - abs(len(sentence) - 190) / 80


def is_candidate(sentence: str) -> bool:
    if not 90 <= len(sentence) <= 420:
        return False
    lowered = sentence.lower()
    if any(x in lowered for x in ("arxiv:", "copyright", "preprint", "all rights reserved")):
        return False
    if lowered.startswith(("figure ", "table ", "references", "appendix")):
        return False
    if len(re.findall(r"[A-Za-z]", sentence)) < 50:
        return False
    return True


def select_questions(pdf_path: Path) -> list[dict[str, str | int]]:
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    target = question_count(page_count)
    per_bin = [[] for _ in range(target)]

    for page_idx, page in enumerate(reader.pages):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            continue
        if len(text) < 300:
            continue
        lowered = text.lower()
        if lowered.count("references") > 3 and page_idx > page_count * 0.65:
            continue

        bin_idx = min(target - 1, math.floor(page_idx / max(page_count, 1) * target))
        for sentence in split_sentences(text):
            if is_candidate(sentence):
                per_bin[bin_idx].append((sentence_score(sentence), page_idx + 1, sentence, text))

    selected = []
    used_answers = set()

    for bin_items in per_bin:
        for _, page_num, sentence, page_text in sorted(bin_items, reverse=True):
            normalized = sentence.lower()
            if normalized in used_answers:
                continue
            used_answers.add(normalized)
            selected.append((page_num, sentence, page_text))
            break

    if len(selected) < target:
        all_items = [item for group in per_bin for item in group]
        for _, page_num, sentence, page_text in sorted(all_items, reverse=True):
            normalized = sentence.lower()
            if normalized in used_answers:
                continue
            used_answers.add(normalized)
            selected.append((page_num, sentence, page_text))
            if len(selected) == target:
                break

    rows = []
    source = pdf_path.name
    paper_name = pdf_path.stem
    for idx, (page_num, answer, page_text) in enumerate(selected[:target], start=1):
        topic = keyword_phrase(answer)
        question = f"According to '{paper_name}', what does the paper state about {topic}?"
        context = page_text[:1600]
        rows.append(
            {
                "pdf": source,
                "page_count": page_count,
                "expected_page": page_num,
                "question": question,
                "reference": answer,
                "reference_context": context,
                "question_type": "extractive",
                "difficulty": "easy" if page_count <= 20 else "medium" if page_count <= 50 else "hard",
            }
        )
    return rows


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    all_rows = []
    row_id = 1
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        for row in select_questions(pdf_path):
            row["id"] = f"ragas-{row_id:04d}"
            all_rows.append(row)
            row_id += 1

    fieldnames = [
        "id",
        "question",
        "reference",
        "reference_context",
        "pdf",
        "expected_page",
        "page_count",
        "question_type",
        "difficulty",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
