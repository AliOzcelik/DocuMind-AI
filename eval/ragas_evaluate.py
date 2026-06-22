from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from rag_chain import build_prompt, get_chat_llm, get_retriever
from router import decide_route


cfg = Config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run this RAG app over a RAGAS dataset.")
    parser.add_argument("--dataset", default="eval/ragas_sample_dataset.csv")
    parser.add_argument("--responses-out", default="")
    parser.add_argument("--scores-out", default="")
    parser.add_argument("--model", default=cfg.llm_model, help="Ollama generator model.")
    parser.add_argument("--evaluator-model", default=cfg.llm_model, help="Ollama model used by RAGAS as judge.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows.")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        help="Comma-separated RAGAS metric names.",
    )
    return parser.parse_args()


def default_output_path(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(ROOT / "eval" / f"{prefix}_{stamp}.csv")


def load_rows(path: str, limit: int) -> list[dict]:
    with open(ROOT / path if not Path(path).is_absolute() else path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def run_rag(question: str, model: str) -> dict:
    docs = get_retriever().invoke(question)
    route = decide_route(docs)
    prompt = build_prompt(question, history=[], docs=docs, route=route)
    response = get_chat_llm(model).invoke(prompt)
    contexts = [doc.page_content for doc in docs]
    sources = [
        f"{doc.metadata.get('source', 'unknown')}#p{doc.metadata.get('page_number', doc.metadata.get('page', '?'))}"
        for doc in docs
    ]
    return {
        "answer": response.content,
        "contexts": contexts,
        "sources": sources,
        "route": route.value,
    }


def write_responses(rows: list[dict], path: str) -> None:
    fields = [
        "id",
        "question",
        "reference",
        "answer",
        "contexts",
        "retrieved_sources",
        "expected_pdf",
        "expected_page",
        "route",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def import_ragas_metrics(metric_names: list[str]):
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ModuleNotFoundError as e:
        raise SystemExit(
            "RAGAS is not installed. Install it first:\n"
            "  venv/bin/pip install ragas datasets\n"
        ) from e

    available = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    unknown = [name for name in metric_names if name not in available]
    if unknown:
        raise SystemExit(f"Unknown metrics: {', '.join(unknown)}")
    return [available[name] for name in metric_names]


def make_ragas_llm(model: str):
    try:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
    except Exception:
        return None

    judge = ChatOllama(model=model, base_url=cfg.ollama_url, temperature=0)
    return LangchainLLMWrapper(judge)


def run_ragas(response_rows: list[dict], metric_names: list[str], evaluator_model: str):
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ModuleNotFoundError as e:
        raise SystemExit(
            "RAGAS dependencies are not installed. Install them first:\n"
            "  venv/bin/pip install ragas datasets\n"
        ) from e

    metrics = import_ragas_metrics(metric_names)
    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"].split("\n---CONTEXT---\n") if row["contexts"] else [],
                "ground_truth": row["reference"],
            }
            for row in response_rows
        ]
    )

    evaluator_llm = make_ragas_llm(evaluator_model)
    kwargs = {"dataset": dataset, "metrics": metrics, "raise_exceptions": False}
    if evaluator_llm is not None:
        kwargs["llm"] = evaluator_llm

    return evaluate(**kwargs)


def main() -> None:
    args = parse_args()
    dataset_rows = load_rows(args.dataset, args.limit)
    responses_out = args.responses_out or default_output_path("ragas_llm_responses")
    scores_out = args.scores_out or default_output_path("ragas_scores")
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    response_rows = []
    print(f"Running RAG on {len(dataset_rows)} questions with model={args.model}")
    for i, row in enumerate(dataset_rows, start=1):
        question = row["question"]
        print(f"[{i}/{len(dataset_rows)}] {row.get('id', '')} {question[:90]}")
        result = run_rag(question, args.model)
        response_rows.append(
            {
                "id": row.get("id", ""),
                "question": question,
                "reference": row.get("reference", ""),
                "answer": result["answer"],
                "contexts": "\n---CONTEXT---\n".join(result["contexts"]),
                "retrieved_sources": " | ".join(result["sources"]),
                "expected_pdf": row.get("pdf", ""),
                "expected_page": row.get("expected_page", ""),
                "route": result["route"],
            }
        )

    write_responses(response_rows, responses_out)
    print(f"\nSaved LLM responses: {responses_out}")

    print(f"\nRunning RAGAS metrics: {', '.join(metric_names)}")
    scores = run_ragas(response_rows, metric_names, args.evaluator_model)
    print("\nRAGAS result:")
    print(scores)

    scores_df = scores.to_pandas()
    scores_df.to_csv(scores_out, index=False)
    print(f"\nSaved RAGAS scores: {scores_out}")


if __name__ == "__main__":
    main()
