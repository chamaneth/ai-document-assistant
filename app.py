import gradio as gr
import tempfile
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from transformers import pipeline

# ----- 1. Embedding model (for searching documents) -----
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----- 2. LLM model (FLAN-T5) -----
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=device,
    max_length=512
)

llm = HuggingFacePipeline(pipeline=pipe)

# ----- 3. Global variables -----
vector_db = None
qa_bot = None

# ----- 4. Process PDF -----
def load_and_process_pdf(pdf_file):
    global vector_db, qa_bot

    if pdf_file is None:
        return "Please upload a PDF first."

    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        path = tmp.name

    # Load PDF
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Create vector store (persistent)
    vector_db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    vector_db.persist()

    # Create RetrievalQA chain
    qa_bot = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    )

    return "PDF processed successfully! Ask your question."

# ----- 5. Chat function -----
def chat_with_pdf(question):
    if qa_bot is None:
        return "Upload a PDF first."

    result = qa_bot.invoke({"query": question})
    return result["result"]

# ----- 6. Gradio UI -----
with gr.Blocks() as demo:
    gr.Markdown("# 📄 AI PDF Document Assistant")

    pdf_input = gr.File(label="Upload PDF")
    status = gr.Textbox(label="Status")

    process_btn = gr.Button("Process PDF")

    question = gr.Textbox(label="Ask Question")
    answer = gr.Textbox(label="Answer")

    process_btn.click(load_and_process_pdf, pdf_input, status)
    question.submit(chat_with_pdf, question, answer)

demo.launch()
