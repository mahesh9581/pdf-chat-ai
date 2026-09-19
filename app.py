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
    layout="wide"
)

# ─── Premium CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #070711;
    color: #E0E0F0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

.user-msg {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 10px 0 10px 40px;
    color: #E0E0FF;
    font-size: 14px;
    line-height: 1.6;
}

.ai-msg {
    background: rgba(15,15,30,0.8);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 10px 40px 10px 0;
    color: #C8C8E0;
    font-size: 14px;
    line-height: 1.6;
}

.user-label {
    font-size: 11px;
    color: #6366F1;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.ai-label {
    font-size: 11px;
    color: #00FFB2;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.pdf-card {
    background: rgba(0,255,178,0.05);
    border: 1px solid rgba(0,255,178,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 16px;
    flex: 1;
    text-align: center;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E0E0F0 !important;
    padding: 12px 16px !important;
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
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.3) !important;
}

hr {
    border-color: rgba(255,255,255,0.06) !important;
}

section[data-testid="stSidebar"] {
    background: #0D0D1A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

section[data-testid="stSidebar"] * {
    color: #E0E0F0 !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #070711; }
::-webkit-scrollbar-thumb { background: #1A1A2E; border-radius: 2px; }

.stFileUploader > div {
    background: rgba(99,102,241,0.05) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    color: #888 !important;
    font-size: 12px !important;
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

# ─── API Key from .env ────────────────────────────────────────
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

Rules:
- Answer ONLY from the provided context
- Be detailed and specific
- List all relevant points you find
- If asked to summarize give a complete summary
- Never say information is unavailable if it exists in context
- Use bullet points for lists

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

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Chat AI")
    st.markdown("---")
    st.markdown("### Upload Document")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            st.session_state.pdf_ready = False
            st.session_state.messages = []
            st.session_state.pdf_name = uploaded_file.name

        if not st.session_state.pdf_ready:
            progress = st.progress(0)
            status = st.empty()

            status.text("📖 Reading PDF...")
            progress.progress(25)
            text, pages = extract_text_from_pdf(uploaded_file)

            status.text("✂️ Creating chunks...")
            progress.progress(50)
            chunks = chunk_text(text)

            status.text("🧠 Building embeddings...")
            progress.progress(75)
            collection = build_vector_store(chunks)

            progress.progress(100)
            status.text("✅ Ready!")

            st.session_state.collection = collection
            st.session_state.pdf_ready = True
            st.session_state.pdf_stats = {
                "name": uploaded_file.name,
                "pages": pages,
                "chunks": len(chunks),
                "chars": len(text)
            }

        if st.session_state.pdf_ready:
            st.markdown("---")
            st.markdown("### 📊 Document Info")
            stats = st.session_state.pdf_stats
            st.markdown(f"""
            <div class='pdf-card'>
                <div style='font-size:12px; color:#00FFB2;
                font-weight:600; margin-bottom:12px;'>
                    ✅ {stats['name'][:30]}
                </div>
                <div style='display:flex; gap:8px;'>
                    <div class='metric-box'>
                        <div style='font-size:20px; font-weight:700;
                        color:#6366F1;'>{stats['pages']}</div>
                        <div style='font-size:10px; color:#555;
                        margin-top:2px;'>Pages</div>
                    </div>
                    <div class='metric-box'>
                        <div style='font-size:20px; font-weight:700;
                        color:#6366F1;'>{stats['chunks']}</div>
                        <div style='font-size:10px; color:#555;
                        margin-top:2px;'>Chunks</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        st.markdown(
            f"**💬 {len(st.session_state.messages)} messages**"
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#444; line-height:2;'>
        <b style='color:#666;'>How it works:</b><br>
        1. Upload any PDF<br>
        2. PDF is chunked and embedded<br>
        3. Ask any question<br>
        4. AI finds relevant chunks<br>
        5. Get accurate answers!
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN AREA ────────────────────────────────────────────────

# Header
st.markdown("""
<div style='text-align:center; padding:20px 0 10px;'>
    <h1 style='font-size:2.5rem; font-weight:800;
    color:#fff; letter-spacing:-1px; margin:0;'>
        PDF <span style='background:linear-gradient(135deg,#6366F1,#00FFB2);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;'>Chat AI</span>
    </h1>
    <p style='color:#555; font-size:15px; margin-top:8px;'>
        Upload any PDF and have an intelligent conversation with it
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Chat messages
if not st.session_state.pdf_ready:
    st.markdown("""
    <div style='text-align:center; padding:80px 20px;'>
        <div style='font-size:72px; margin-bottom:20px;'>📄</div>
        <h3 style='color:#fff; font-weight:700; margin-bottom:10px;'>
            No Document Loaded
        </h3>
        <p style='color:#555; font-size:14px;
        max-width:400px; margin:0 auto;'>
            Upload a PDF from the sidebar to start chatting.
            Ask questions, get summaries, extract information!
        </p>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.messages:
    st.markdown(f"""
    <div style='text-align:center; padding:40px 20px;'>
        <div style='font-size:56px; margin-bottom:16px;'>🚀</div>
        <h3 style='color:#fff; font-weight:700; margin-bottom:8px;'>
            Document Ready!
        </h3>
        <p style='color:#555; font-size:14px;'>
            Ask anything about
            <b style='color:#6366F1;'>
            {st.session_state.pdf_stats.get('name', 'your document')}
            </b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-bottom:16px;'>
        <span style='font-size:11px; color:#333;
        letter-spacing:2px; text-transform:uppercase;'>
            Suggested Questions
        </span>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "Summarize this document",
        "What are the main topics?",
        "List all important points",
        "What technologies are mentioned?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, key=f"sug_{i}"):
                st.session_state.auto_question = s
                st.rerun()

else:
    for msg in st.session_state.messages:
        st.markdown(f"""
        <div class='user-label'>You</div>
        <div class='user-msg'>{msg['question']}</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='ai-label'>AI Assistant</div>
        <div class='ai-msg'>{msg['answer']}</div>
        """, unsafe_allow_html=True)

        if 'chunks' in msg:
            with st.expander("📚 View source chunks"):
                for j, chunk in enumerate(msg['chunks']):
                    st.markdown(f"**Chunk {j+1}:**")
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06);
                    border-radius:8px; padding:12px;
                    font-size:12px; color:#888; line-height:1.6;'>
                        {chunk}
                    </div>
                    """, unsafe_allow_html=True)

# ─── Input Area ───────────────────────────────────────────────
st.markdown("---")

if st.session_state.pdf_ready:
    default_q = ""
    if hasattr(st.session_state, 'auto_question'):
        default_q = st.session_state.auto_question
        del st.session_state.auto_question

    col1, col2 = st.columns([5, 1])

    with col1:
        question = st.text_input(
            "question",
            placeholder="Ask anything about your document...",
            label_visibility="collapsed",
            value=default_q,
            key="q_input"
        )

    with col2:
        ask = st.button("Send 🚀", use_container_width=True)

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

else:
    st.markdown("""
    <div style='text-align:center; padding:20px;
    background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px;'>
        <span style='color:#444; font-size:14px;'>
            👈 Upload a PDF from the sidebar to start
        </span>
    </div>
    """, unsafe_allow_html=True)