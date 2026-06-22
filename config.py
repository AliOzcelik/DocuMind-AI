from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Config:
    pdf_dir: Path = Path("data/pdf")
    chroma_dir: Path = Path("chroma_db")
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:4b-mlx"
    #embed_model: str = "nomic-embed-text"
    embed_model: str = "qwen3-embedding:0.6b"
    collection_name: str = "papers"
    candidate_k: int = 20      # per-leg retrieval pool (BM25 + dense) before reranking
    rerank_top_k: int = 5      # final chunks kept after the cross-encoder
    chunk_size: int = 1000
    chunk_overlap: int = 200
    temperature: float = 1.0
    relevance_threshold: float = 0.3
    
    # collection_metadata: dict = {"hnsw:space": "cosine"}
    collection_metadata: dict = field(default_factory=lambda: {"hnsw:space": "cosine"})
    
    
    
    
# cfg = Config()