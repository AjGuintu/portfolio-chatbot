# create_database.py
import os
from pathlib import Path
from typing import List
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenAIEmbeddingVectorStore  # new
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
VECTOR_STORE_NAME = os.environ.get("VECTOR_STORE_NAME", "portfolio_vector_db")

def list_documents() -> List[str]:
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    return [
        f for f in os.listdir(DATA_PATH)
        if f.lower().endswith((".txt", ".md", ".docx"))
    ]

def _load_file_as_documents(path: str, filename: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    text = ""
    if ext in [".md", ".txt"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".docx":
        import docx
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": filename})]

def generate_data_store() -> bool:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    docs = []
    for fname in list_documents():
        docs.extend(_load_file_as_documents(os.path.join(DATA_PATH, fname), fname))

    if not docs:
        print("No documents to index.")
        return True

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vecstore = OpenAIEmbeddingVectorStore.from_documents(
        documents=chunks,
        embedding=embed_model,
        vector_store_id=VECTOR_STORE_NAME
    )
    print(f"Indexed {len(chunks)} documents into vector store {VECTOR_STORE_NAME}")
    return True
