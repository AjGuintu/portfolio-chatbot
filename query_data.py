import os
import openai
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import OpenAI
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_NAME = os.environ.get("VECTOR_STORE_NAME", "portfolio_vector_db")

def get_db():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return OpenAI(
        embedding=embeddings,
        index_name=VECTOR_STORE_NAME
    )

def debug_search(text: str):
    db = get_db()
    res = db.similarity_search_with_relevance_scores(text, k=5)
    return [{"source": d.metadata.get("source", "unknown"), "score": float(score)} for d, score in res]

def query_function(question: str):
    db = get_db()
    results = db.similarity_search_with_relevance_scores(question, k=4)
    filtered = [(doc, score) for doc, score in results if score >= 0.15]

    if not filtered:
        return "I don't know. Please upload related documents.", []

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])
    hf_pipe = pipeline("text2text-generation", model="google/flan-t5-small")
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    prompt = f"Answer concisely using ONLY the context below.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    response = llm.invoke(prompt)
    answer_text = response if isinstance(response, str) else response.text

    sources = list({doc.metadata.get("source", "unknown") for doc, _ in filtered})
    return answer_text, sources
