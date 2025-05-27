import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# UI
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄🤖 Chat with your PDF (Groq-Powered LLM)")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Groq LLM
    llm = ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_key="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",
        openai_api_base="https://api.groq.com/openai/v1"
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # User input
    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        with st.spinner("Generating answer..."):
            # Run query
            result = qa_chain(user_input)

            # Append user input and bot response to chat history
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("Bot", result["result"]))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        if sender == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")
