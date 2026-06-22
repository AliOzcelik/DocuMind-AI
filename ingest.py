from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import pypdf
import os
from config import Config

cfg = Config()
folder_path = cfg.pdf_dir


def load_pdf_documents(folder_path):
    filenames = [f.name for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(".pdf")]

    documents = []

    for filename in filenames:
        try:
            reader = pypdf.PdfReader(os.path.join(folder_path, filename))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text, metadata={"source": filename, "page_number": reader.pages.index(page) + 1}))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    return documents


def ingest_pdf_documents(documents, 
                         chunk_size = cfg.chunk_size, 
                         chunk_overlap = cfg.chunk_overlap, 
                         embedding_model = cfg.embed_model, 
                         ollama_url = cfg.ollama_url,
                         chroma_directory = cfg.chroma_dir,
                         collection_metadata = cfg.collection_metadata):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, 
                                                   chunk_overlap = chunk_overlap)
    
    chunks = text_splitter.split_documents(documents)
    
    embedding_model = OllamaEmbeddings(model = embedding_model,
                                       base_url = ollama_url)

    vector_store = Chroma.from_documents(documents = chunks,
                                         embedding = embedding_model,
                                         persist_directory = str(chroma_directory),
                                         collection_name = cfg.collection_name,
                                         collection_metadata = collection_metadata)
    
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    
    
def load_and_split_pdf(path):
    reader = pypdf.PdfReader(path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(path), "page_number": i + 1},
            ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    return splitter.split_documents(documents)

