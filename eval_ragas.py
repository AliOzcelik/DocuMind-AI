import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from rag_chain import answer, get_retriever

rows = pd.read_csv("eval/questions.csv")

records = []
for _, row in rows.iterrows():
    question = row["question"]
    reference = row.get("reference", "")

    docs = get_retriever().invoke(question)
    result = answer(question, history=[])

    records.append({
        "question": question,
        "answer": result.text,
        "contexts": [doc.page_content for doc in docs],
        "ground_truth": reference,
    })

dataset = Dataset.from_list(records)

scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print(scores)
scores.to_pandas().to_csv("eval/ragas_results.csv", index=False)