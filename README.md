# DocuMind AI - Hybrid RAG Chat System

DocuMind AI is a local Retrieval-Augmented Generation (RAG) chat application for asking questions over a PDF library. It combines keyword retrieval, dense vector retrieval, cross-encoder reranking, route selection, multi-turn chat memory, model switching through Ollama, and a browser-based chat UI.

The system is designed for research-paper question answering, but the same architecture can be used for any PDF corpus.

## Outputs
<p>
  <img src="images/image.png" width="45%">
  <img src="images/app.png" width="45%">
</p>

## Features

- PDF ingestion into a persistent Chroma vector database
- Recursive text chunking with configurable chunk size and overlap
- Ollama embeddings for dense retrieval
- BM25 keyword retrieval
- Hybrid retrieval through LangChain `EnsembleRetriever`
- Cross-encoder reranking with relevance scores
- Automatic routing between document-grounded RAG answers and general answers
- Streaming chat responses through FastAPI Server-Sent Events
- Multi-turn conversation memory within each chat
- Chat history persistence with SQLite
- Runtime LLM selection from `ollama list`
- Frontend PDF upload and source display
- RAGAS evaluation dataset and runner under `eval/`

## Repository Structure

```text
RAG_system/
├── data/pdf/                 # Main PDF corpus
├── ragas_data/               # PDFs used for RAGAS sample dataset generation
├── chroma_db/                # Persisted Chroma vector database
├── frontend/                 # Static browser UI
├── images/                   # Logo and avatar assets
├── eval/                     # RAGAS dataset, evaluation script, guide
├── main.py                   # FastAPI app and HTTP/SSE endpoints
├── ingest.py                 # PDF loading, chunking, and Chroma ingestion
├── vectorstore.py            # Chroma and Ollama embedding helpers
├── retriever.py              # BM25 + dense hybrid retrieval + reranking
├── router.py                 # RAG vs general route decision
├── rag_chain.py              # Prompt construction and LLM calls
├── chat.py                   # Conversation memory wrapper
├── database.py               # SQLite chat persistence
├── config.py                 # Models, paths, chunking, retrieval settings
└── requirements.txt
```

## Architecture

```mermaid
flowchart LR
    User["User"] --> UI["Frontend\nHTML/CSS/JS"]
    UI --> API["FastAPI Backend\nmain.py"]

    API --> Chat["ChatSession\nchat.py"]
    Chat --> RAG["RAG Chain\nrag_chain.py"]

    RAG --> Retriever["Hybrid Retriever\nretriever.py"]
    Retriever --> BM25["BM25 keyword search"]
    Retriever --> Dense["Chroma dense search"]
    Dense --> Emb["Ollama Embeddings"]
    Retriever --> Rerank["Cross-Encoder Reranker"]

    Rerank --> Router["Route Decision\nrouter.py"]
    Router --> Prompt["Prompt Builder\nRAG or General"]
    Prompt --> LLM["Selected Ollama LLM"]
    LLM --> API
    API --> UI

    API <--> DB["SQLite\nchats.db"]
    API --> Models["ollama list\nmodel selector"]
```

## Data Flow

```mermaid
flowchart TD
    PDFs["PDF files\n(data/pdf or upload)"] --> Load["Extract page text\npypdf"]
    Load --> Split["Split into chunks\nRecursiveCharacterTextSplitter"]
    Split --> Embed["Embed chunks\nOllamaEmbeddings"]
    Embed --> Chroma["Persist chunks + vectors\nChromaDB"]

    Question["User question"] --> Hybrid["Hybrid retrieval"]
    Chroma --> DenseSearch["Dense vector search"]
    Split --> BM25Index["BM25 corpus"]
    BM25Index --> KeywordSearch["Keyword search"]
    DenseSearch --> Hybrid
    KeywordSearch --> Hybrid

    Hybrid --> Rerank["Cross-encoder reranking"]
    Rerank --> TopDocs["Top-k context chunks\nwith source/page metadata"]
    TopDocs --> Route["Route: RAG or General"]

    Route --> Prompt["Build prompt\ncurrent question + chat history"]
    TopDocs --> Prompt
    History["Recent chat history"] --> Prompt
    Prompt --> Ollama["Ollama chat model"]
    Ollama --> Answer["Answer + route + sources"]
```

## Input / Output Flow

```mermaid
flowchart LR
    subgraph Inputs
        Q["query"]
        SID["session_id"]
        M["selected model"]
        PDF["optional uploaded PDF"]
    end

    subgraph Processing
        Upload["/upload\nsave + split + embed"]
        Stream["/chat/stream"]
        Retrieve["retrieve + rerank"]
        Decide["decide route"]
        Generate["generate answer"]
        Persist["persist messages"]
    end

    subgraph Outputs
        Tokens["streamed answer tokens"]
        Meta["route + sources + model"]
        Chats["saved chat history"]
        Context["source cards in UI"]
    end

    PDF --> Upload
    Q --> Stream
    SID --> Stream
    M --> Stream
    Stream --> Retrieve --> Decide --> Generate
    Generate --> Tokens
    Decide --> Meta
    Retrieve --> Context
    Stream --> Persist --> Chats
```

## Retrieval Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as FastAPI
    participant R as Retriever
    participant C as Chroma
    participant B as BM25
    participant X as Cross-Encoder
    participant L as Ollama LLM
    participant D as SQLite

    U->>F: Ask question
    F->>A: POST /chat/stream {query, session_id, model}
    A->>D: Save user message
    A->>R: Retrieve relevant chunks
    R->>B: Keyword search
    R->>C: Dense vector search
    B-->>R: Candidate chunks
    C-->>R: Candidate chunks
    R->>X: Rerank candidates
    X-->>A: Top context chunks + scores
    A->>L: Prompt with question, history, and context if RAG
    L-->>A: Stream tokens
    A-->>F: SSE events
    A->>D: Save assistant response
    F-->>U: Render answer and sources
```

## Prerequisites

- Python 3.11+
- Ollama installed and running
- At least one chat model installed in Ollama
- One embedding model installed in Ollama

Example Ollama models:

```bash
ollama pull qwen3.5:4b-mlx
ollama pull qwen3-embedding:0.6b
```

Start Ollama:

```bash
ollama serve
```

## Installation

```bash
git clone <your-repo-url>
cd RAG_system

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Main settings are in `config.py`:

```python
pdf_dir = Path("data/pdf")
chroma_dir = Path("chroma_db")
ollama_url = "http://localhost:11434"
llm_model = "qwen3.5:4b-mlx"
embed_model = "qwen3-embedding:0.6b"
chunk_size = 1000
chunk_overlap = 200
candidate_k = 20
rerank_top_k = 5
relevance_threshold = 0.3
```

Change these values before ingestion if you want a different embedding model, chunking strategy, or Chroma collection.

## Ingest PDFs

Place PDF files in:

```text
data/pdf/
```

Then start the app. If the retriever is not ready, startup attempts ingestion automatically. You can also call:

```bash
curl -X POST http://127.0.0.1:5000/ingest
```

Uploaded PDFs from the frontend are saved into `data/pdf/`, split, embedded, and added to Chroma.

## Run the App

```bash
uvicorn main:app --host 127.0.0.1 --port 5000 --reload
```

Open:

```text
http://127.0.0.1:5000
```

## API Endpoints

| Endpoint | Method | Purpose |
|---|---:|---|
| `/` | GET | Serves the frontend |
| `/health` | GET | Health check and configured models |
| `/models` | GET | Lists available Ollama chat models, sorted alphabetically |
| `/chat` | POST | Non-streaming chat response |
| `/chat/stream` | POST | Streaming chat response through SSE |
| `/upload` | POST | Upload and ingest a PDF |
| `/ingest` | POST | Ingest all PDFs from `data/pdf` |
| `/chats` | GET | List saved chats |
| `/chats/{chat_id}` | GET | Get messages for one chat |
| `/chats/{chat_id}` | DELETE | Delete a chat |

Example chat request:

```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is LoRA?",
    "session_id": "demo",
    "model": "qwen3.5:4b-mlx"
  }'
```

Example response:

```json
{
  "answer": "LoRA is a parameter-efficient fine-tuning method...",
  "route": "rag",
  "sources": [
    {
      "source": "LORA, Low Rank Adaptation of LLMs.pdf",
      "page": 1,
      "score": 0.82
    }
  ]
}
```

## Model Switching

The frontend calls:

```text
GET /models
```

The backend reads `ollama list`, filters out embedding models, sorts chat models alphabetically, and returns them to the model selector. Each chat request includes the selected model, so you can switch LLMs within the same conversation.

## Chat Memory

Each chat has a `ChatSession` with recent conversation history. The system keeps the latest messages and includes them in prompt construction for both RAG and general responses. Saved chats are persisted in SQLite and rehydrated into memory when opened again.

## RAGAS Evaluation

Evaluation assets live under `eval/`:

```text
eval/
├── build_ragas_dataset.py
├── ragas_sample_dataset.csv
├── ragas_evaluate.py
└── RAGAS.md
```

Install RAGAS dependencies if needed:

```bash
venv/bin/pip install ragas datasets
```

Run a smoke test:

```bash
venv/bin/python eval/ragas_evaluate.py --limit 5
```

Run the full evaluation:

```bash
venv/bin/python eval/ragas_evaluate.py
```

The script saves:

- LLM answers for each question
- retrieved contexts and sources
- row-level RAGAS scores
- aggregate RAGAS output printed in the terminal

See `eval/RAGAS.md` for hyperparameter experiments such as BM25/dense weights, chunk size, chunk overlap, and semantic chunking.

## Important Notes

- `chroma_db/` is generated data. Do not edit it manually.
- Changing `chunk_size`, `chunk_overlap`, embedding model, or collection name requires re-ingestion.
- Ollama must be running for embedding, chat, and model listing.
- The model selector intentionally excludes models with `embed` in the name.
- RAGAS datasets should be evaluated against the same PDF corpus that was indexed into Chroma.

## License

Add your preferred license before publishing this repository.
