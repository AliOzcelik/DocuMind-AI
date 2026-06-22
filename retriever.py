from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.documents import Document

from vectorstore import get_vector_store, get_embeddings
from config import Config

cfg = Config()

def load_corpus():
    embedding_model = get_embeddings(cfg.embed_model, cfg.ollama_url)
    vector_store = get_vector_store(str(cfg.chroma_dir), cfg.collection_name, embedding_model)

    all_docs = vector_store.get()
    documents = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    return documents, vector_store


def get_bm25_vectorstore():
    documents, vector_store = load_corpus()
    bm25_retriever = BM25Retriever.from_documents(documents, k=cfg.candidate_k)
    return bm25_retriever, vector_store


def get_ensemble_retriever(keywords_weight=0.5, semantic_weight=0.5):
    bm25_retriever, vector_store = get_bm25_vectorstore()
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": cfg.candidate_k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[keywords_weight, semantic_weight],
    )
    return hybrid_retriever


class ScoringReranker(CrossEncoderReranker):
    def compress_documents(self, documents, query, callbacks=None):
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        kept = []
        for doc, score in ranked[: self.top_n]:
            doc.metadata["relevance_score"] = float(score)
            kept.append(doc)
        return kept


def get_reranker():
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    reranker = ScoringReranker(model=model, top_n=cfg.rerank_top_k)
    return reranker


def get_hybrid_retriever(keywords_weight=0.5, semantic_weight=0.5):
    ensemble = get_ensemble_retriever(keywords_weight=keywords_weight,
                                      semantic_weight=semantic_weight)
    reranker = get_reranker()

    hybrid_retriever = ContextualCompressionRetriever(
        base_retriever = ensemble,
        base_compressor = reranker,
    )
    return hybrid_retriever
