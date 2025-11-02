import streamlit as st
from query_data import query_function, debug_search
from create_database import generate_data_store, list_documents
import os

DATA_PATH = "data/"

os.makedirs(DATA_PATH, exist_ok=True)

st.set_page_config(page_title="Portfolio Chatbot 🤖", layout="wide")
st.title("📘 Portfolio RAG Chatbot")
st.write("Ask anything about your uploaded knowledge base.")

# ---------------- CHAT AREA ----------------
st.subheader("💬 Ask a Question")
user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Thinking..."):
        answer, sources = query_function(user_input)

    st.markdown(f"### 🤖 Answer:\n{answer}")

    if sources:
        st.write("📌 Sources:")
        for src in sources:
            st.code(src)
    else:
        st.warning("⚠️ No relevant sources found.")

# ---------------- UPLOAD AREA ----------------
st.subheader("📂 Upload Files (.txt / .md / .docx)")
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "md", "docx"],
    accept_multiple_files=True,
)

auto_rebuild = st.checkbox("Auto rebuild database", value=True)

if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join(DATA_PATH, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s)!")

    if auto_rebuild:
        with st.spinner("Rebuilding knowledge database..."):
            generate_data_store()
            st.success("✅ Database updated!")
            st.experimental_rerun()

# ---------------- DEBUG PANEL ----------------
with st.expander("🛠 Debug Info"):
    st.write("📄 Stored documents:")
    docs = list_documents()
    st.code("\n".join(docs) if docs else "No files found")

    if st.button("Test Search: keyword = 'test'"):
        debug_results = debug_search("test")
        st.write(debug_results)
