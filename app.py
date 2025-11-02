# app.py
import os
import streamlit as st
from create_database import generate_data_store, list_documents
from query_data import query_function, debug_search

DATA_PATH = "data/"
os.makedirs(DATA_PATH, exist_ok=True)

st.set_page_config(page_title="Portfolio Chatbot 🤖", layout="wide")

st.title("📘 Portfolio RAG Chatbot")
st.write("Upload .txt/.md/.docx files, rebuild index, ask questions.")

# Chat section
st.subheader("💬 Ask a question")
user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Searching..."):
        answer, sources = query_function(user_input)
    st.markdown("### 🤖 Answer")
    st.write(answer)
    if sources:
        st.write("📌 Sources:")
        for s in sources:
            st.code(s)
    else:
        st.info("No relevant sources found.")

st.markdown("---")

# Upload files
st.subheader("📂 Upload files (.txt / .md / .docx)")
uploaded = st.file_uploader("Choose files", type=["txt", "md", "docx"], accept_multiple_files=True)
auto_rebuild = st.checkbox("Auto rebuild after upload", True)

if uploaded:
    for f in uploaded:
        path = os.path.join(DATA_PATH, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Uploaded {len(uploaded)} file(s).")
    if auto_rebuild:
        with st.spinner("Rebuilding index..."):
            if generate_data_store():
                st.success("✅ Index rebuilt. Ready to ask questions!")
            else:
                st.error("❌ Rebuild failed. Check logs.")

# Manage documents
st.subheader("🗑 Delete a file")
docs = list_documents()
if docs:
    choice = st.selectbox("Select file to remove", docs)
    if st.button("Delete"):
        try:
            os.remove(os.path.join(DATA_PATH, choice))
            st.success(f"Deleted {choice}. Please rebuild index to remove it from memory.")
        except Exception as e:
            st.error(f"Error deleting: {e}")
else:
    st.info("No files uploaded yet.")

# Manual rebuild button
if st.button("🔄 Rebuild Knowledge DB"):
    with st.spinner("Rebuilding index..."):
        if generate_data_store():
            st.success("✔️ Rebuild complete!")
        else:
            st.error("❌ Rebuild failed.")

# Debug info
with st.expander("🛠 Debug Info"):
    st.write("Files in data/:")
    st.write(list_documents())
    if st.button("Test search 'test'"):
        st.write(debug_search("test"))
