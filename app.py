import streamlit as st
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="📄",
    layout="centered"
)

# ─── Premium CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #070711; color: #E0E0F0; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main .block-container {
    padding: 1.5rem 1.5rem;
    max-width: 800px;
}

.user-msg {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0 8px 30px;
    color: #E0E0FF;
    font-size: 14px;
    line-height: 1.6;
}

.ai-msg {
    background: rgba(15,15,30,0.8);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 30px 8px 0;
    color: #C8C8E0;
    font-size: 14px;
    line-height: 1.6;
}

.user-label {
    font-size: 10px;
    color: #6366F1;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
    margin-top: 16px;
}

.ai-label {
    font-size: 10px;
    color: #00FFB2;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
    margin-top: 16px;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #E0E0F0 !important;
    padding: 14px 16px !important;
    font-size: 14px !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.3) !important;
}

.stFileUploader > div {
    background: rgba(99,102,241,0.05) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
    padding: 10px !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #6366F1, #00FFB2) !important;
    border-radius: 4px !important;
}

hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 16px 0 !important;
}

.metric-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
}

[data-testid="stMetricValue"] {
    color: #6366F1 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #555 !important;
    font-size: 11px !important;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 8px !important;
    color: #555 !important;
    font-size: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #070711; }
::-webkit-scrollbar-thumb { background: #1A1A2E; border-radius: 2px; }

/* Mobile responsive */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 1rem;
    }
    .user-msg { margin-left: 10px; }
    .ai-msg { margin-right: 10px; }
}
</style>
""", unsafe_allow_html=True)

# ─── Initialize Models ────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma():
    return chromadb.Client()

embedding_model = load_embedding_model()
chroma_client = get_chroma()

# ─── API Key ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── Session State ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = {}

# ─── Functions ────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file):
    text = ""
    pages = 0
    with pdfplumber.open(pdf_file) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text, pages

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_text(text)

def build_vector_store(chunks):
    try:
        chroma_client.delete_collection("pdf_store")
    except:
        pass
    collection = chroma_client.create_collection("pdf_store")
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"c{i}" for i in range(len(chunks))]
    )
    return collection

def search_chunks(collection, query, n=5):
    query_emb = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=n
    )
    return results['documents'][0]

def generate_answer(question, chunks, history):
    context = "\n\n---\n\n".join(chunks)

    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for h in history[-4:]:
            history_text += f"User: {h['question']}\n"
            history_text += f"Assistant: {h['answer']}\n\n"

    prompt = f"""You are an intelligent document assistant.
Answer questions accurately based ONLY on the document content.
Be detailed and specific. List all relevant points.
Never say information is unavailable if it exists in context.
Use bullet points for lists.

{history_text}
Document Context:
{context}

User Question: {question}

Detailed Answer:"""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1
    )
    return response.choices[0].message.content

# ─── UI ───────────────────────────────────────────────────────

# Header
st.markdown("""
<div style='text-align:center; padding:24px 0 8px;'>
    <h1 style='font-size:2rem; font-weight:800;
    color:#fff; letter-spacing:-1px; margin:0;'>
        📄 PDF <span style='background:linear-gradient(
        135deg,#6366F1,#00FFB2);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;'>
        Chat AI</span>
    </h1>
    <p style='color:#444; font-size:13px; margin-top:8px;'>
        Upload any PDF and chat with it using AI
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Upload Section ───────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📄 Upload your PDF here",
    type=["pdf"],
    help="Upload any PDF document to start chatting"
)

if uploaded_file:
    # Detect new PDF
    if uploaded_file.name != st.session_state.pdf_name:
        st.session_state.pdf_ready = False
        st.session_state.messages = []
        st.session_state.pdf_name = uploaded_file.name

    # Process PDF
    if not st.session_state.pdf_ready:
        progress = st.progress(0)
        status = st.empty()

        status.text("📖 Reading PDF...")
        progress.progress(20)
        text, pages = extract_text_from_pdf(uploaded_file)

        status.text("✂️ Creating chunks...")
        progress.progress(45)
        chunks = chunk_text(text)

        status.text("🧠 Building embeddings...")
        progress.progress(75)
        collection = build_vector_store(chunks)

        progress.progress(100)
        status.empty()
        progress.empty()

        st.session_state.collection = collection
        st.session_state.pdf_ready = True
        st.session_state.pdf_stats = {
            "name": uploaded_file.name,
            "pages": pages,
            "chunks": len(chunks),
        }
        st.success("✅ PDF processed and ready to chat!")

    # Show PDF stats
    if st.session_state.pdf_ready:
        stats = st.session_state.pdf_stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document",
                stats['name'][:12] + "..." if len(stats['name']) > 12
                else stats['name'])
        with col2:
            st.metric("📑 Pages", stats['pages'])
        with col3:
            st.metric("🧩 Chunks", stats['chunks'])

st.divider()

# ─── Chat Section ─────────────────────────────────────────────
if not st.session_state.pdf_ready:
    # Empty state
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <div style='font-size:56px; margin-bottom:16px;'>📄</div>
        <h3 style='color:#fff; font-weight:700;
        margin-bottom:8px; font-size:1.3rem;'>
            No Document Loaded
        </h3>
        <p style='color:#444; font-size:14px;
        max-width:300px; margin:0 auto;'>
            Upload a PDF above to start
            an intelligent conversation!
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Clear button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Chat messages
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding:40px 20px;'>
            <div style='font-size:44px; margin-bottom:14px;'>
                💬
            </div>
            <p style='color:#444; font-size:14px;'>
                Ask anything about your document!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            st.markdown(
                f"<div class='user-label'>You</div>"
                f"<div class='user-msg'>{msg['question']}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='ai-label'>AI Assistant</div>"
                f"<div class='ai-msg'>{msg['answer']}</div>",
                unsafe_allow_html=True
            )
            if 'chunks' in msg:
                with st.expander("📚 View source chunks"):
                    for j, chunk in enumerate(msg['chunks']):
                        st.markdown(f"**Chunk {j+1}:**")
                        st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.06);
                        border-radius:8px; padding:10px;
                        font-size:12px; color:#777;
                        line-height:1.6; margin-bottom:8px;'>
                            {chunk}
                        </div>
                        """, unsafe_allow_html=True)

    st.divider()

    # Input area
    question = st.text_input(
        "question",
        placeholder="Ask anything about your PDF...",
        label_visibility="collapsed",
        key="q_input"
    )

    ask = st.button("🚀 Send Message", use_container_width=True)

    if ask and question:
        with st.spinner("🔍 Searching document..."):
            chunks = search_chunks(
                st.session_state.collection,
                question
            )
        with st.spinner("🤖 Generating answer..."):
            answer = generate_answer(
                question,
                chunks,
                st.session_state.messages
            )
        st.session_state.messages.append({
            "question": question,
            "answer": answer,
            "chunks": chunks
        })
        st.rerun()

    elif ask and not question:
        st.warning("Please type a question first!")