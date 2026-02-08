# 📄 AI Document Assistant

![Status: Work in Progress](https://img.shields.io/badge/status-Work%20in%20Progress-orange?style=flat&logo=github)

AI Document Assistant is a local AI-powered tool that lets you upload PDFs and interact with them using natural language queries. It uses embeddings and language models to process documents and answer questions based on their content.

> ⚠️ **Note:** HuggingFace dependencies currently may cause conflicts with `transformers` and `tokenizers`.  
> If you encounter errors, you can use the **previous stable version of the code available on GitHub**.

---

## ⚡ Features

- Upload PDF documents and process them for Q&A
- Ask questions about PDF content in natural language
- Uses **LangChain**, **ChromaDB**, and **HuggingFace** models for embeddings and text generation
- Interactive web UI built with **Gradio**

---

## 🛠 Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd ai-document-assistant
