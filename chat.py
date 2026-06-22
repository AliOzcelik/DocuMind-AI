from langchain_core.messages import HumanMessage, AIMessage

from rag_chain import answer, answer_stream, Answer
from router import Route


EXIT_COMMANDS = {"exit", "quit", "/bye"}
RESET_COMMAND = "/reset"
MAX_HISTORY_MESSAGES = 10   # keep the last ~5 turns (move to config if you like)


class ChatSession:
    def __init__(self):
        self.history: list = []   # HumanMessage / AIMessage objects

    def load_history(self, messages):
        self.history = []
        for message in messages:
            role = getattr(message, "role", None)
            content = getattr(message, "content", "")
            if role == "user":
                self.history.append(HumanMessage(content=content))
            elif role == "assistant":
                self.history.append(AIMessage(content=content))
        self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def ask(self, query: str, model_name: str | None = None) -> Answer:
        """One turn: answer with current history, then record both messages."""
        result = answer(query, self.history, model_name)

        # Append AFTER answering, then trim to bound the context window.
        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=result.text))
        self.history = self.history[-MAX_HISTORY_MESSAGES:]

        return result

    def ask_stream(self, query: str, model_name: str | None = None):
        """Stream one turn: yield events from answer_stream, accumulate the full
        text, then record both messages to history when the stream finishes."""
        full = ""
        for evt in answer_stream(query, self.history, model_name):
            if evt.get("type") == "token":
                full += evt["text"]
            yield evt
        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=full))
        self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def reset(self):
        self.history.clear()


def print_answer(result: Answer):
    tag = "from documents" if result.route == Route.RAG else "general knowledge"
    print(f"\nAssistant [{tag}]:\n{result.text}")

    if result.route == Route.RAG and result.sources:
        print("\nSources:")
        seen = set()
        for s in result.sources:
            key = (s["source"], s["page"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {s['source']} (page {s['page']})")
    print()


def run_chat():
    print("RAG chatbot ready. Ask a question — '/reset' clears memory, 'exit' quits.\n")
    session = ChatSession()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            break
        if query.lower() == RESET_COMMAND:
            session.reset()
            print("Conversation memory cleared.\n")
            continue

        result = session.ask(query)
        print_answer(result)
