from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from config import Config
from router import Route, decide_route
from retriever import get_hybrid_retriever

cfg = Config()


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a collection of "
    "research papers. Answer ONLY using the context provided below. If the "
    "answer is not in the context, say you don't know — do not invent anything. "
    "Cite the source file and page number for each claim you make."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user's question is not covered by the "
    "available documents, so answer from your own general knowledge, clearly "
    "and concisely."
)

@dataclass
class Answer:
    text: str
    route: Route
    sources: list[dict] = field(default_factory=list)


def get_llm(llm_model, ollama_url, temperature):
    
    llm = ChatOllama(model = llm_model, 
                     url = ollama_url,
                     temperature = temperature)
    
    return llm


retriever = None  # cached hybrid retriever


# Cache the hybrid retriever — rebuilding BM25 from Chroma every turn is slow.
def get_retriever():
    global retriever
    if retriever is None:
        retriever = get_hybrid_retriever()
    return retriever


# Drop the cached retriever so the next get_retriever() rebuilds it.
# Call after adding new documents so BM25 picks up the new chunks.
def refresh_retriever():
    global retriever
    retriever = None



# docs: list[Document]
"""
def format_context(docs):
    if not docs:
        return "No relevant documents found."
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        page_number = doc.metadata.get("page_number", "unknown page")
        content = doc.page_content
        formatted_docs.append(f"Source: {source}, Page: {page_number}\n{content}")
    return "\n\n".join(formatted_docs)
"""


# Turn retrieved chunks into one context string, each chunk tagged with its source + page so the model can cite. Helper for the RAG prompt.
def format_context(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", doc.metadata.get("page", "?"))
        blocks.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_history(history) -> str:
    turns = []
    for message in history:
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        else:
            role = getattr(message, "type", "Message").title()
        turns.append(f"{role}: {getattr(message, 'content', str(message))}")
    return "\n".join(turns)


def build_prompt(query, history, docs, route):
    history_text = format_history(history)
    is_rag = route == Route.RAG or route == Route.RAG.value

    if is_rag:
        context = format_context(docs)
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n"
            "Use the conversation history to understand follow-up questions, "
            "but ground factual claims in the document context.\n\n"
            f"Context:\n{context}\n\n"
        )
    else:
        prompt = (
            f"{GENERAL_SYSTEM_PROMPT}\n"
            "Use the conversation history to understand follow-up questions "
            "and keep the answer consistent with this chat.\n\n"
        )

    if history_text:
        prompt += f"Conversation History:\n{history_text}\n\n"

    prompt += f"Current Question: {query}\nAnswer:"
    return prompt


llms = {}  # cached chat models by Ollama model name


def get_chat_llm(model_name: str | None = None):
    model = (model_name or cfg.llm_model).strip()
    if model not in llms:
        llms[model] = get_llm(model, cfg.ollama_url, cfg.temperature)
    return llms[model]


def answer(query, history, model_name: str | None = None):
    docs = get_retriever().invoke(query)
    route = decide_route(docs)
    prompt = build_prompt(query, history, docs, route)
    response = get_chat_llm(model_name).invoke(prompt)

    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page_number", d.metadata.get("page", "?")),
            "score": d.metadata.get("relevance_score"),
        }
        for d in docs
    ]

    return Answer(text=response.content, route=route, sources=sources)


def answer_stream(query, history, model_name: str | None = None):
    docs = get_retriever().invoke(query)
    route = decide_route(docs)
    prompt = build_prompt(query, history, docs, route)

    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page_number", d.metadata.get("page", "?")),
            "score": d.metadata.get("relevance_score"),
        }
        for d in docs
    ]
    yield {"type": "meta", "route": route.value, "sources": sources, "model": (model_name or cfg.llm_model).strip()}

    for chunk in get_chat_llm(model_name).stream(prompt):
        text = getattr(chunk, "content", "") or ""
        if text:
            yield {"type": "token", "text": text}
