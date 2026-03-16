import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") # type: ignore


llm = ChatGroq( 
    model="llama-3.3-70b-versatile"
)

#Upload PDF files

st.header("📄 RAG Powered PDF Question Answering Chatbot")


#Sidebar
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")


#Extract : IF file Uploaded
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # st.write(text)

    #Breaking  it into chunks
    # Step 2: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],
        chunk_size = 1000,
        chunk_overlap = 150,
        # it take last 150 character to make the next chunks meaningful one
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Generating Embeddings
    # Step 3: Create Embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5")

    #CREATING a vector store FAISS
    # Step 4: Store embeddings in FAISS
    vector_store = FAISS.from_texts(chunks,embeddings)

    # Step 5: Convert vector store to retriever
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    # Step 6: Create Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )

    # User input
    user_question = st.text_input("Ask a question from the PDF")

    #step 7 : Generate Answer

    if user_question:

        response = qa_chain.invoke({"query":user_question})

        st.subheader("Answer")
        st.write(response["result"])
