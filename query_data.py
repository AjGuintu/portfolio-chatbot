# query_data.py
import os
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

INDEX_NAME = os.environ.get("PINECONE_INDEX", "portfolio-rag")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _init_pinecone():
    api_key = os.environ.get("PINECONE_API_KEY")
    env = os.environ.get("PINECONE_ENV")
    if not api_key or not env:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV environment variables.")
    pinecone.init(api_key=api_key, environment=env)

def get_db():
    """Return a LangChain Pinecone vectorstore connected to your index."""
    _init_pinecone()
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # use existing index
    try:
        return Pinecone.from_existing_index(embedding, index_name=INDEX_NAME)
    except Exception:
        # fallback: try building a Pinecone wrapper
        return Pinecone(embedding_function=embedding, index_name=INDEX_NAME)

def debug_search(text: str):
    try:
        db = get_db()
        results = db.similarity_search_with_relevance_scores(text, k=5)
        out = []
        for doc, score in results:
            out.append({"source": doc.metadata.get("source", "unknown"), "score": float(score)})
        return out
    except Exception as e:
        return {"error": str(e)}

def query_function(query_text: str):
    try:
        db = get_db()
    except Exception as e:
        return f"Error connecting to Pinecone: {e}", []

    # retrieve
    results = db.similarity_search_with_relevance_scores(query_text, k=4)

    # filter low-scoring results (tune threshold as needed)
    filtered = [(doc, score) for doc, score in results if score >= 0.15]
    if not filtered:
        return "I don't know. Please upload related documents.", []

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])

    # generate a concise answer using local Flan-T5 or other LLM
    try:
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        prompt = f"Answer concisely using ONLY the context below.\n\nCONTEXT:\n{context}\n\nQUESTION: {query_text}\nANSWER:"
        response = llm.invoke(prompt)
        text = response if isinstance(response, str) else getattr(response, "text", str(response))
    except Exception as e:
        # fallback: return context snippet if LLM failed
        print("LLM error:", e)
        text = context[:1000]  # abridge

    sources = list({doc.metadata.get("source", "unknown") for doc, _ in filtered})
    return text, sources
