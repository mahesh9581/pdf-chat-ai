import streamlit as st
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp { background: #06060F; }

#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0C0C1A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    width: 320px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 24px 20px !important;
}

/* Main area */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Upload area */
.stFileUploader {
    background: rgba(99,102,241,0.04) !important;
}
.stFileUploader > div {
    border: 2px dashed rgba(99,102,241,0.35) !important;
    border-radius: 14px !important;
    background: rgba(99,102,241,0.04) !important;
    padding: 8px !important;
}
.stFileUploader label {
    color: #888 !important;
    font-size: 13px !important;
}

/* Input */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #E0E0F0 !important;
    padding: 14px 18px !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    background: rgba(99,102,241,0.06) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #333 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 13px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.25s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(99,102,241,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Clear button override */
.clear-btn > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #666 !important;
    padding: 8px 16px !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}
.clear-btn > button:hover {
    background: rgba(255,77,109,0.08) !important;
    border-color: rgba(255,77,109,0.2) !important;
    color: #FF6B85 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366F1, #00FFB2) !important;
    border-radius: 4px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    color: #6366F1 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #555 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    color: #555 !important;
    font-size: 12px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1A1A2E; border-radius: 4px; }

/* Spinner */
.stSpinner > div { border-top-color: #6366F1 !important; }

/* Success */
.stSuccess {
    background: rgba(0,229,160,0.06) !important;
    border: 1px solid rgba(0,229,160,0.2) !important;
    border-radius: 10px !important;
    color: #00E5A0 !important;
}

/* Warning */
.stWarning {
    background: rgba(255,184,0,0.06) !important;
    border: 1px solid rgba(255,184,0,0.2) !important;
    border-radius: 10px !important;
}

/* Mobile */
@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Initialize ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma():
    return chromadb.Client()

model = load_model()
chroma_client = get_chroma()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── Session State ────────────────────────────────────────────
for key, val in {
    "messages": [],
    "collection": None,
    "pdf_name": "",
    "pdf_ready": False,
    "pdf_stats": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Functions ────────────────────────────────────────────────
def extract_text(pdf_file):
    text = ""
    pages = 0
    with pdfplumber.open(pdf_file) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text, pages

def chunk_text(text):
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    ).split_text(text)

def build_store(chunks):
    try:
        chroma_client.delete_collection("pdf_store")
    except:
        pass
    col = chroma_client.create_collection("pdf_store")
    col.add(
        documents=chunks,
        embeddings=model.encode(chunks).tolist(),
        ids=[f"c{i}" for i in range(len(chunks))]
    )
    return col

def search(collection, query, n=5):
    res = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=n
    )
    return res['documents'][0]

def answer(question, chunks, history):
    context = "\n\n---\n\n".join(chunks)
    hist = ""
    if history:
        hist = "Previous conversation:\n"
        for h in history[-4:]:
            hist += f"User: {h['question']}\nAssistant: {h['answer']}\n\n"

    prompt = f"""You are an intelligent document assistant.
Answer questions accurately based ONLY on the document content.
Be detailed and specific. List all relevant points.
Never say information is unavailable if it exists in context.
Use bullet points for lists.

{hist}
Document Context:
{context}

User Question: {question}

Detailed Answer:"""

    res = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1
    )
    return res.choices[0].message.content

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:

    # Logo
    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div style='font-size:22px; font-weight:800; color:#fff;
        letter-spacing:-0.5px;'>
            📄 PDF <span style='background:linear-gradient(
            135deg,#6366F1,#00FFB2);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;'>Chat AI</span>
        </div>
        <div style='font-size:11px; color:#333; margin-top:4px;
        letter-spacing:1px; text-transform:uppercase;'>
            Powered by RAG Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("""
    <div style='font-size:11px; color:#555; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:10px; font-weight:600;'>
        Upload Document
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:
            st.session_state.pdf_ready = False
            st.session_state.messages = []
            st.session_state.pdf_name = uploaded.name

        if not st.session_state.pdf_ready:
            prog = st.progress(0)
            stat = st.empty()

            stat.caption("📖 Reading PDF...")
            prog.progress(20)
            text, pages = extract_text(uploaded)

            stat.caption("✂️ Creating chunks...")
            prog.progress(50)
            chunks = chunk_text(text)

            stat.caption("🧠 Building embeddings...")
            prog.progress(80)
            col = build_store(chunks)

            prog.progress(100)
            stat.empty()
            prog.empty()

            st.session_state.collection = col
            st.session_state.pdf_ready = True
            st.session_state.pdf_stats = {
                "name": uploaded.name,
                "pages": pages,
                "chunks": len(chunks),
                "chars": len(text)
            }

        # PDF Info
        if st.session_state.pdf_ready:
            st.markdown("<div style='height:16px'></div>",
                unsafe_allow_html=True)

            # File name
            st.markdown(f"""
            <div style='background:rgba(0,229,160,0.06);
            border:1px solid rgba(0,229,160,0.18);
            border-radius:10px; padding:12px 14px;
            margin-bottom:14px;'>
                <div style='font-size:10px; color:#00E5A0;
                letter-spacing:1.5px; text-transform:uppercase;
                margin-bottom:5px; font-weight:600;'>
                    ✅ Ready
                </div>
                <div style='font-size:13px; color:#ccc;
                word-break:break-all;'>
                    {st.session_state.pdf_stats['name']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Stats
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Pages",
                    st.session_state.pdf_stats['pages'])
            with c2:
                st.metric("Chunks",
                    st.session_state.pdf_stats['chunks'])

    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.02);
        border:1px solid rgba(255,255,255,0.05);
        border-radius:12px; padding:20px;
        text-align:center; margin-top:8px;'>
            <div style='font-size:32px; margin-bottom:10px;'>📄</div>
            <div style='color:#444; font-size:13px;
            line-height:1.6;'>
                Upload a PDF to start chatting with it
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>",
        unsafe_allow_html=True)
    st.divider()

    # Chat stats
    if st.session_state.messages:
        st.markdown(f"""
        <div style='font-size:12px; color:#444;
        margin-bottom:12px;'>
            💬 {len(st.session_state.messages)} messages
        </div>
        """, unsafe_allow_html=True)

    # Clear button
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History",
        use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # How it works
    st.markdown("""
    <div style='font-size:11px; color:#333;
    line-height:2; letter-spacing:0.3px;'>
        <div style='color:#555; font-weight:600;
        margin-bottom:6px; font-size:11px;
        text-transform:uppercase; letter-spacing:1.5px;'>
            How It Works
        </div>
        <div>📤 Upload any PDF</div>
        <div>✂️ Text is chunked smartly</div>
        <div>🧠 Chunks stored as embeddings</div>
        <div>🔍 Semantic search finds answers</div>
        <div>🤖 AI generates accurate response</div>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN AREA ────────────────────────────────────────────────
main = st.container()

with main:
    # Top bar
    st.markdown("""
    <div style='padding:24px 32px 0;
    border-bottom:1px solid rgba(255,255,255,0.05);
    margin-bottom:0; display:flex;
    align-items:center; justify-content:space-between;'>
        <div>
            <div style='font-size:18px; font-weight:700;
            color:#fff; letter-spacing:-0.3px;'>
                Conversation
            </div>
            <div style='font-size:12px; color:#333;
            margin-top:3px;'>
                Ask anything about your document
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat area
    chat = st.container()

    with chat:
        st.markdown("""
        <div style='padding:0 32px; min-height:60vh;'>
        """, unsafe_allow_html=True)

        if not st.session_state.pdf_ready:
            # Empty state
            st.markdown("""
            <div style='display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            padding:100px 20px; text-align:center;'>
                <div style='width:80px; height:80px;
                background:rgba(99,102,241,0.08);
                border:1px solid rgba(99,102,241,0.15);
                border-radius:20px; display:flex;
                align-items:center; justify-content:center;
                font-size:36px; margin-bottom:20px;'>
                    📄
                </div>
                <div style='font-size:20px; font-weight:700;
                color:#fff; margin-bottom:10px;'>
                    No Document Loaded
                </div>
                <div style='font-size:14px; color:#444;
                max-width:320px; line-height:1.6;'>
                    Upload a PDF from the sidebar to start
                    an intelligent conversation with your document
                </div>
                <div style='margin-top:28px; display:flex;
                gap:8px; flex-wrap:wrap; justify-content:center;'>
                    <span style='font-size:11px; color:#2A2A3E;
                    background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.06);
                    padding:6px 14px; border-radius:20px;'>
                        📑 Any PDF
                    </span>
                    <span style='font-size:11px; color:#2A2A3E;
                    background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.06);
                    padding:6px 14px; border-radius:20px;'>
                        🔍 Semantic Search
                    </span>
                    <span style='font-size:11px; color:#2A2A3E;
                    background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.06);
                    padding:6px 14px; border-radius:20px;'>
                        🤖 AI Powered
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.messages:
            # PDF ready but no messages
            name = st.session_state.pdf_stats.get('name', 'document')
            st.markdown(f"""
            <div style='display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            padding:80px 20px; text-align:center;'>
                <div style='width:72px; height:72px;
                background:rgba(0,229,160,0.08);
                border:1px solid rgba(0,229,160,0.2);
                border-radius:18px; display:flex;
                align-items:center; justify-content:center;
                font-size:32px; margin-bottom:18px;'>
                    ✅
                </div>
                <div style='font-size:18px; font-weight:700;
                color:#fff; margin-bottom:8px;'>
                    Document Ready!
                </div>
                <div style='font-size:13px; color:#444;
                margin-bottom:28px;'>
                    <span style='color:#6366F1;'>{name}</span>
                    is loaded and ready
                </div>
                <div style='font-size:12px; color:#333;
                margin-bottom:16px; letter-spacing:1px;
                text-transform:uppercase;'>
                    Try asking
                </div>
                <div style='display:flex; flex-direction:column;
                gap:8px; width:100%; max-width:360px;'>
            """, unsafe_allow_html=True)

            suggestions = [
                "📋 Summarize this document",
                "🎯 What are the main topics?",
                "💡 List all key points",
                "🛠 What skills are mentioned?",
            ]

            for sug in suggestions:
                if st.button(sug, key=f"sug_{sug}",
                    use_container_width=True):
                    st.session_state.auto_q = sug.split(" ", 1)[1]
                    st.rerun()

            st.markdown("</div></div>", unsafe_allow_html=True)

        else:
            # Show messages
            st.markdown("<div style='padding-top:20px;'>",
                unsafe_allow_html=True)

            for msg in st.session_state.messages:
                # User message
                st.markdown(f"""
                <div style='display:flex; justify-content:flex-end;
                margin-bottom:4px;'>
                    <div style='font-size:10px; color:#6366F1;
                    font-weight:600; letter-spacing:1px;
                    text-transform:uppercase;'>You</div>
                </div>
                <div style='background:rgba(99,102,241,0.12);
                border:1px solid rgba(99,102,241,0.2);
                border-radius:16px 16px 4px 16px;
                padding:14px 18px;
                margin:0 0 16px 60px;
                color:#E0E0FF; font-size:14px;
                line-height:1.65;'>
                    {msg['question']}
                </div>
                """, unsafe_allow_html=True)

                # AI message
                st.markdown(f"""
                <div style='display:flex;
                justify-content:flex-start;
                margin-bottom:4px;'>
                    <div style='font-size:10px; color:#00E5A0;
                    font-weight:600; letter-spacing:1px;
                    text-transform:uppercase;'>
                        🤖 AI Assistant
                    </div>
                </div>
                <div style='background:rgba(12,12,26,0.9);
                border:1px solid rgba(255,255,255,0.07);
                border-radius:16px 16px 16px 4px;
                padding:14px 18px;
                margin:0 60px 8px 0;
                color:#C8C8E0; font-size:14px;
                line-height:1.65;'>
                    {msg['answer']}
                </div>
                """, unsafe_allow_html=True)

                if 'chunks' in msg:
                    with st.expander(
                        "📚 View source chunks used"):
                        for j, chunk in enumerate(msg['chunks']):
                            st.markdown(f"""
                            <div style='background:rgba(
                            255,255,255,0.02);
                            border:1px solid rgba(255,255,255,0.06);
                            border-radius:8px; padding:12px;
                            font-size:12px; color:#666;
                            line-height:1.65; margin-bottom:8px;'>
                                <span style='color:#444;
                                font-size:10px; font-weight:600;
                                text-transform:uppercase;
                                letter-spacing:1px;'>
                                    Chunk {j+1}
                                </span><br><br>
                                {chunk}
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ─── Input Bar ────────────────────────────────────────────
    st.markdown("""
    <div style='position:sticky; bottom:0;
    background:linear-gradient(0deg,
    rgba(6,6,15,1) 0%, rgba(6,6,15,0.95) 100%);
    padding:16px 32px 24px; margin-top:16px;
    border-top:1px solid rgba(255,255,255,0.05);'>
    """, unsafe_allow_html=True)

    if st.session_state.pdf_ready:
        # Handle auto question from suggestions
        default_q = ""
        if hasattr(st.session_state, 'auto_q'):
            default_q = st.session_state.auto_q
            del st.session_state.auto_q

        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input(
                "q",
                value=default_q,
                placeholder="Ask anything about your document...",
                label_visibility="collapsed",
                key="main_input"
            )
        with c2:
            send = st.button("Send 🚀",
                use_container_width=True)

        if send and question:
            with st.spinner("🔍 Searching..."):
                chunks = search(
                    st.session_state.collection,
                    question
                )
            with st.spinner("🤖 Generating..."):
                ans = answer(
                    question,
                    chunks,
                    st.session_state.messages
                )
            st.session_state.messages.append({
                "question": question,
                "answer": ans,
                "chunks": chunks
            })
            st.rerun()

        elif send and not question:
            st.warning("Please type a question!")

    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.02);
        border:1px solid rgba(255,255,255,0.05);
        border-radius:14px; padding:16px 20px;
        text-align:center; color:#333; font-size:13px;'>
            👈 Upload a PDF from the sidebar to start chatting
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)