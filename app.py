# app.py
import os
import streamlit as st
from create_database import generate_data_store, list_documents
from query_data import query_function, debug_search

DATA_PATH = "data/"
os.makedirs(DATA_PATH, exist_ok=True)

st.set_page_config(page_title="Portfolio Chatbot (Pinecone)", layout="wide")
st.title("📘 Portfolio RAG Chatbot (Pinecone)")
st.write("Upload .txt/.md/.docx files, rebuild the index, and ask questions.")

# Chat
st.subheader("💬 Ask a question")
q = st.text_input("Your question:")
if q:
    with st.spinner("Searching..."):
        answer, sources = query_function(q)
    st.markdown("### 🤖 Answer")
    st.write(answer)
    if sources:
        st.write("📌 Sources:")
        for s in sources:
            st.code(s)
    else:
        st.info("No relevant sources found.")

st.markdown("---")

# Upload
st.subheader("📂 Upload files (.txt / .md / .docx)")
files = st.file_uploader("Choose files", type=["txt", "md", "docx"], accept_multiple_files=True)
auto_rebuild = st.checkbox("Auto rebuild after upload", True)

if files:
    for f in files:
        with open(os.path.join(DATA_PATH, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Uploaded {len(files)} file(s).")
    if auto_rebuild:
        with st.spinner("Rebuilding index..."):
            ok = generate_data_store()
            if ok:
                st.success("Index rebuilt.")
            else:
                st.error("Index rebuild failed. Check logs.")

# Delete file (removes file only — you'll manually rebuild per your B preference)
st.subheader("🗑 Delete a file")
docs = list_documents()
if docs:
    pick = st.selectbox("Select file to remove (this deletes file only)", docs)
    if st.button("Delete file"):
        try:
            os.remove(os.path.join(DATA_PATH, pick))
            st.success(f"Deleted {pick}. Note: Run 'Rebuild Knowledge DB' to remove its vectors from Pinecone.")
        except Exception as e:
            st.error(f"Could not delete: {e}")
else:
    st.info("No files in data/.")

# Rebuild button
if st.button("🔄 Rebuild Knowledge DB"):
    with st.spinner("Rebuilding Pinecone index..."):
        ok = generate_data_store()
        if ok:
            st.success("Rebuild complete.")
        else:
            st.error("Rebuild failed. Check console or logs.")

# Debug panel
with st.expander("🛠 Debug"):
    st.write("Local files:")
    st.write(list_documents())
    if st.button("Test search 'test'"):
        st.write(debug_search("test"))
