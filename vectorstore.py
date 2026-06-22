from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma




def get_embeddings(model_name, ollama_url):
    embedding_model = OllamaEmbeddings(model=model_name, base_url=ollama_url)
    return embedding_model


def create_vector_store(documents, embedding_model, persist_directory, collection_metadata):
    
    vector_store = Chroma.from_documents(documents=documents, 
                                         embedding=embedding_model, 
                                         persist_directory=persist_directory,
                                         collection_metadata=collection_metadata)
    return vector_store


def get_vector_store(persist_directory, collection_name, embedding_function):
    vector_store = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    return vector_store


# store = get_vector_store(str(cfg.chroma_dir), cfg.collection_name, get_embeddings())

def get_retriever(vector_store, search_kwargs):
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return retriever