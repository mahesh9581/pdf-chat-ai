# 📄 PDF Chat AI

An intelligent document chatbot built with RAG (Retrieval Augmented Generation) pipeline.
Upload any PDF and ask questions — get accurate answers instantly!

## 🚀 Live Demo
[Click here to try it live](https://pdf-chat-ai-gfefsumaxz8hm46ia4eumf.streamlit.app/)

## ✨ Features
- Upload any PDF document
- Ask questions in natural language
- AI finds relevant chunks using semantic search
- Accurate answers using Groq LLM
- Chat history maintained
- Premium dark UI

## 🛠 Tech Stack
| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| LLM | Groq AI |
| PDF Processing | pdfplumber |
| Text Splitting | LangChain |

## 🧠 How it Works
1. PDF is uploaded and text is extracted
2. Text is split into chunks of 1000 characters
3. Chunks are converted to embeddings using sentence-transformers
4. Embeddings are stored in ChromaDB vector database
5. User asks a question
6. Question is converted to embedding
7. ChromaDB finds most similar chunks using cosine similarity
8. Groq LLM generates accurate answer from those chunks

## ⚙️ Run Locally

### 1. Clone the repo
git clone https://github.com/mahesh9581/pdf-chat-ai.git
cd pdf-chat-ai

### 2. Install dependencies
pip install -r requirements.txt

### 3. Create .env file
GROQ_API_KEY=your_groq_key_here

### 4. Run
streamlit run app.py

## 📦 Requirements
- Python 3.8+
- Groq API Key (free at console.groq.com)
