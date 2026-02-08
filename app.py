import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA 
from transformers import pipeline

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pipe = pipeline(
    "text-generation",
    model="google/flan-t5-small",   # small = faster
    max_length=512
)

llm = HuggingFacePipeline(pipeline=pipe)

vector_db = None
qa_bot = None


def load_and_process_pdf(pdf_file):
    global vector_db, qa_bot

    if pdf_file is None:
        return "Please upload a PDF first."

    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vector_db = Chroma.from_documents(chunks, embeddings)

    qa_bot = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever()
    )

    return "PDF processed successfully! Ask your question."


# ----- Chat function -----
def chat_with_pdf(question):
    if qa_bot is None:
        return "Upload a PDF first."

    return qa_bot.run(question)


# ----- UI -----
with gr.Blocks() as demo:
    gr.Markdown("# 📄 AI Document Assistant")

    pdf_input = gr.File(label="Upload PDF")
    status = gr.Textbox(label="Status")

    process_btn = gr.Button("Process PDF")

    question = gr.Textbox(label="Ask Question")
    answer = gr.Textbox(label="Answer")

    process_btn.click(load_and_process_pdf, pdf_input, status)
    question.submit(chat_with_pdf, question, answer)


demo.launch()
