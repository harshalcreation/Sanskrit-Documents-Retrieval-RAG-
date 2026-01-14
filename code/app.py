import streamlit as st
import re
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

load_dotenv()

st.set_page_config(
    page_title="Sanskrit Document Retrieval",
    layout="wide"
)

st.title("📜 Sanskrit Document Retrieval (RAG)")
st.write("**Extractive, CPU-based Sanskrit Document Question Answering**")

if "query_memory" not in st.session_state:
    st.session_state.query_memory = []


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0900-\u097F\u0020-\u007E।॥]", "", text)
    return text.strip()

def is_valid_sanskrit_query(text: str, min_chars: int = 3) -> bool:
    chars = re.findall(r"[\u0900-\u097F]", text)
    non_sanskrit = re.sub(r"[\u0900-\u097F\s।॥?]", "", text)
    return len(chars) >= min_chars and non_sanskrit.strip() == ""


@st.cache_resource(show_spinner=True)
def build_retriever():
    BASE_DIR = Path(__file__).resolve().parent.parent
    PDF_PATH = BASE_DIR / "data" / "Rag.pdf"

    loader = UnstructuredPDFLoader(
        str(PDF_PATH),
        mode="elements"
)
    raw_documents = loader.load()

    cleaned_docs = []
    for doc in raw_documents:
        cleaned = clean_text(doc.page_content)
        if len(cleaned) > 50:
            doc.page_content = cleaned
            cleaned_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["॥", "।", "\n\n", "\n", " "]
    )

    documents = splitter.split_documents(cleaned_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever, len(documents)


with st.spinner("🔧 Initializing Sanskrit RAG system..."):
    retriever, total_chunks = build_retriever()

st.success(f"✅ Index built successfully ({total_chunks} chunks)")



with st.sidebar:
    st.subheader("🧠 Query Memory (स्मृतिः)")

    if st.session_state.query_memory:
        with st.expander("📜 पूर्व प्रश्नाः (Click to view)", expanded=False):
            for i, q in enumerate(reversed(st.session_state.query_memory[-10:]), 1):
                st.markdown(f"{i}. {q}")

            if st.button("Clear Memory"):
                st.session_state.query_memory.clear()
                st.rerun()
    else:
        st.caption("स्मृतिः रिक्ता अस्ति।")

query = st.text_input(
    "🔍 संस्कृत प्रश्न प्रविष्टं कुर्वन्तु:",
    placeholder="उदा: मूर्खभृत्यस्य कथायाः उपदेशः कः?"
)

retrieve_btn = st.button("📖 सन्दर्भं अन्वेषयतु (Retrieve)",use_container_width=True)

if retrieve_btn:

    if not query.strip():
        st.warning("⚠️ कृपया प्रथमं प्रश्नं प्रविश्यताम्।")
        st.stop()

    if not is_valid_sanskrit_query(query):
        st.warning("⚠️ कृपया केवलं संस्कृतभाषायां (देवनागरीलिप्या) प्रश्नं प्रविश्यताम्।")
        st.stop()

    if query not in st.session_state.query_memory:
        st.session_state.query_memory.append(query)

    with st.spinner("📖 सन्दर्भं अन्विष्यते..."):
        docs = retriever.invoke(query)
        docs = [d for d in docs if len(d.page_content.strip()) > 10]

    st.subheader("📌 उत्तर (सन्दर्भात्)")

    if not docs:
        st.warning("दत्तसन्दर्भे उत्तरं न उपलब्धम्।")
    else:
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**सन्दर्भ {i}:**")
            st.write(doc.page_content)
            st.markdown("---")
