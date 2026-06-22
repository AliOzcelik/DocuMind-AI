from enum import Enum
from langchain_core.documents import Document
from config import Config

cfg = Config()


class Route(str, Enum):
    RAG = "rag"
    GENERAL = "general"


# docs: list[Document]

# Helper: pull docs[0].metadata['relevance_score'], or None if docs is empty.
# (CrossEncoderReranker writes this score into each doc's metadata.)
def top_relevance_score(docs):
    if not docs:
        return None
    return docs[0].metadata.get("relevance_score")


def decide_route(docs):
    score = top_relevance_score(docs)
    if score is None:
        return Route.GENERAL
    if score >= cfg.relevance_threshold:
        return Route.RAG
    return Route.GENERAL
