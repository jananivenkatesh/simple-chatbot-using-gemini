import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore

#setup
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# one global in‑memory vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embedding=embeddings)

def clear_vectorstore():
    # recreate empty store
    st.session_state.vector_store = InMemoryVectorStore(embedding=embeddings)

def index_pdf(file_path: Path) -> int:
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    st.session_state.vector_store.add_documents(chunks)
    return len(chunks)

def build_rag_chain():
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that uses ONLY the provided context.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer in a concise way.
"""
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


#streamlit UI
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Chat with your PDF")
st.caption("Upload a PDF in the sidebar, then ask questions in the chat box below.")

# Sidebar
with st.sidebar:
    st.header("Document & Settings")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Clear index", use_container_width=True):
        clear_vectorstore()
        st.session_state.chat_history = []
        st.success("In‑memory index and chat cleared.")

    if uploaded_file is not None and st.button("Process PDF", type="primary", use_container_width=True):
        with st.status("Indexing document...", expanded=True) as status:
            save_path = UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            num_chunks = index_pdf(save_path)
            status.write(f"Indexed {num_chunks} chunks from **{uploaded_file.name}**.")
            status.update(label="Indexing complete!", state="complete")
        st.success("Ready to chat with your document.")

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (bottom text box)
user_input = st.chat_input("Ask a question about the uploaded document...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Answer from RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_chain = build_rag_chain()
            answer = rag_chain.invoke(user_input)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
