# Sanskrit Document Retrieval (RAG)

## 📌 Project Overview
This project implements an **Extractive Retrieval-Augmented Generation (RAG)** system for **Sanskrit documents** using FAISS and HuggingFace embeddings.  
The system answers user queries strictly based on retrieved document context, ensuring **no hallucination**.

---

## 🚀 Features
- Sanskrit-only query validation (Devanagari)
- Extractive question answering
- CPU-based FAISS retrieval
- Query memory with sidebar display
- Clean Streamlit UI
- No LLM-based generation (hallucination-free)

---

## 🗂 Project Structure

RAG_Sanskrit_HarshalBorkar/
├── code/ # Streamlit application
├── data/ # Sanskrit PDF document
├── report/ # Final project report (PDF)
├── README.md # Instructions
└── requirements.txt


---



## ⚙️ Installation

```bash
git clone <repository-url>
cd RAG_Sanskrit_HarshalBorkar
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
