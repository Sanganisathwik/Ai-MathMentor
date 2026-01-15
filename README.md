# ⚡ AI Math Mentor

AI Math Mentor is a high-performance, multimodal mathematical reasoning engine designed to solve complex JEE-level problems. It leverages a multi-agent orchestration workflow powered by **LangGraph**, **Groq (Llama 3.3 70B)**, and **RAG (Retrieval-Augmented Generation)** to provide accurate, step-by-step explanations with professional tutoring methodology.

---

## 🚀 Key Features

- **🎙️ Multimodal Input**: Supports Text, Image (OCR), and Voice (ASR) problem entry.
- **🧠 Multi-Agent Orchestration**: Specialized agents for Parsing, Solving, Verifying, and Explaining problems.
- **📚 Knowledge-Aware (RAG)**: Integrates a local knowledge base to handle specific mathematical domains accurately.
- **✅ Automated Verification**: A dedicated Verifier agent cross-checks logic before presenting a solution.
- **🎙️ Direct Voice Interaction**: Real-time microphone recording and transcription using OpenAI Whisper.
- **💎 Premium UI**: Sleek, glassmorphic dashboard built with Streamlit for an elite user experience.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, LangGraph, Groq SDK
- **Frontend**: Streamlit
- **AI/ML**: 
  - **LLM**: Groq (Llama-3.3-70b-versatile)
  - **ASR**: OpenAI Whisper (via `openai-whisper`)
  - **OCR**: Tesseract OCR
- **Data**: FAISS / TF-IDF for RAG, Pydantic for schemas

---

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **FFmpeg**: Required for audio processing (ASR).
- **Tesseract OCR**: Required for image processing (OCR).

### External Dependencies Installation (Windows)
```powershell
# Install FFmpeg
winget install "FFmpeg (Essentials Build)"

# Install Tesseract
winget install Tesseract-OCR
```

---

## ⚙️ Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Ai-MathMentor.git
   cd Ai-MathMentor
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   *(See `.env.example` for all optional variables)*

---

## 🏃 Running the Application

### 1. Start the Backend (FastAPI)
```powershell
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Frontend (Streamlit)
```powershell
.\.venv\Scripts\streamlit run frontend/app.py
```

---

## 📁 Project Structure

```text
Ai-MathMentor/
├── .env                 # API Keys (Protected)
├── .gitignore           # Git ignore rules
├── requirements.txt     # Global dependencies
├── backend/
│   ├── agents/          # LLM-based logic for Parse/Solve/Verify/Explain
│   ├── api/             # FastAPI Endpoint & Schema definitions
│   ├── data/            # Local data storage for uploads and logs
│   ├── graph/           # LangGraph workflow orchestration
│   ├── llm/             # Groq API client integration
│   ├── rag/             # Knowledge base retrieval (RAG)
│   └── tools/           # Multimodal tools (OCR, ASR)
└── frontend/
    ├── app.py           # Main Streamlit UI
    ├── services/        # Backend API communication
    └── styles/          # CSS for premium UI aesthetics
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
