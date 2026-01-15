import streamlit as st
from services.backend_client import solve_problem, send_feedback, process_image, process_audio
from audiorecorder import audiorecorder
import tempfile
import os
import shutil
import platform

# --- FFmpeg Path Fix for Windows (Auto-Detection) ---
if platform.system() == "Windows":
    if not shutil.which("ffmpeg"):
        winget_path = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
        if os.path.exists(winget_path):
            import glob
            # Search specifically for ffmpeg.exe
            matches = glob.glob(os.path.join(winget_path, "Gyan.FFmpeg*", "**", "ffmpeg.exe"), recursive=True)
            if matches:
                ff_dir = os.path.dirname(matches[0])
                os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
                print(f"--- [FRONTEND] FFmpeg path detected and injected: {ff_dir} ---")
                
                # Help pydub/audiorecorder explicitly
                from pydub import AudioSegment
                AudioSegment.converter = matches[0]
                # Also try to find ffprobe
                ffprobe_matches = glob.glob(os.path.join(ff_dir, "ffprobe.exe"), recursive=False)
                if ffprobe_matches:
                    AudioSegment.ffprobe = ffprobe_matches[0]
                    print(f"--- [FRONTEND] Pydub paths set: {AudioSegment.converter} / {AudioSegment.ffprobe} ---")
            else:
                print("--- [FRONTEND] FFmpeg binary not found in WinGet packages ---")
        else:
            print("--- [FRONTEND] WinGet packages folder not found ---")

# Set page config
st.set_page_config(
    page_title="Math Mentor AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- MODERN UI INJECTION ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #6366f1;
        --secondary: #0ea5e9;
        --bg-dark: #020617;
        --card-bg: rgba(30, 41, 59, 0.5);
    }

    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0, transparent 50%), 
            radial-gradient(at 100% 100%, rgba(14, 165, 233, 0.1) 0, transparent 50%);
    }

    /* Modern Card Style */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }

    /* Typography */
    h1, h2, h3, p, span {
        font-family: 'Inter', sans-serif !important;
    }

    .math-text {
        font-family: 'JetBrains Mono', monospace !important;
        color: #38bdf8;
    }

    /* Custom Input Styling */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
    }

    /* Buttons - Gradient Pulse */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }

    /* Tabs/Radio Styling */
    .stRadio [role="radiogroup"] {
        background: rgba(15, 23, 42, 0.5);
        padding: 5px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Success Result Box */
    .result-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 20px;
    }

    /* Hide redundant elements */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
col_logo, col_status = st.columns([3, 1])
with col_logo:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <div style="font-size: 32px;">🧠</div>
            <div>
                <h2 style="margin: 0; background: linear-gradient(to right, #fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Math Mentor <span style="color:#6366f1">AI</span></h2>
                <p style="margin: 0; font-size: 0.8rem; color: #64748b;">AGENTIC REASONING ENGINE</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_status:
    st.markdown("""
        <div style="text-align: right; margin-top: 10px;">
            <span style="background: rgba(99, 102, 241, 0.1); color: #818cf8; padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.2);">
                ● SYSTEM READY
            </span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- MAIN INTERFACE ---
left_panel, right_panel = st.columns([1, 1.2], gap="large")

with left_panel:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📝 Input Problem")
    
    mode = st.radio(
        "Choose Entry Method",
        ["Text", "Image", "Voice"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    input_query = ""
    
    if mode == "Text":
        input_query = st.text_area("Equation", height=200, placeholder="e.g. Solve for x: 2x² + 5x - 3 = 0", label_visibility="collapsed")
    
    elif mode == "Image":
        uploaded_img = st.file_uploader("Upload Problem Image", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            st.image(uploaded_img, width='stretch', caption="Source Image")
            if st.button("✨ Extract Problem"):
                with st.spinner("OCR Processing..."):
                    res = process_image(uploaded_img)
                    st.session_state.extracted = res.get("extracted_text", "")
        input_query = st.text_area("Extracted Text", value=st.session_state.get('extracted', ''), height=100)

    elif mode == "Voice":
        st.write("Record or upload your problem:")
        audio = []
        try:
            audio = audiorecorder("🎤 Click to Record", "⏹ Stop Recording")
        except Exception as e:
            st.error("🛑 **Audio Recording Unavailable**: FFmpeg/ffprobe not found on your system.")
            st.info("To fix this, run: `winget install ffmpeg` in your terminal and restart the app.")
        
        uploaded_voice = st.file_uploader("Or Upload Audio File", type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "aac", "wma"])
        st.info("💡 **Note**: Microphone access requires browser permissions. If you are using a remote server, ensure you are using HTTPS.")
        
        audio_to_process = None
        
        if len(audio) > 0:
            # Live recording takes precedence
            st.audio(audio.export().read())
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                audio.export(f.name, format="wav")
                audio_to_process = f.name
        elif uploaded_voice:
            st.audio(uploaded_voice)
            audio_to_process = uploaded_voice

        if audio_to_process:
            if st.button("🎙️ Transcribe"):
                with st.spinner("Analyzing audio..."):
                    try:
                        # If it's a filepath from tempfile
                        if isinstance(audio_to_process, str):
                            size = os.path.getsize(audio_to_process)
                            print(f"--- [FRONTEND] Sending file path: {audio_to_process} ({size} bytes) ---")
                            with open(audio_to_process, "rb") as f:
                                res = process_audio(f)
                        else:
                            print(f"--- [FRONTEND] Sending uploaded file: {audio_to_process.name} ({audio_to_process.size} bytes) ---")
                            res = process_audio(audio_to_process)
                        
                        st.session_state.voice_txt = res.get("extracted_text", "")
                    except Exception as e:
                        st.error(f"🛑 **Transcription Failed**: {str(e)}")
                        if "500" in str(e):
                            st.warning("The server encountered an error during audio processing. This is often due to missing FFmpeg or corrupted audio data.")
                        res = {}
        
        input_query = st.text_area("Transcript", value=st.session_state.get('voice_txt', ''), height=100)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 SOLVE WITH GROQ", width='stretch'):
        if input_query.strip():
            with st.spinner("Agent thinking..."):
                res = solve_problem(input_query)
                st.session_state.result = res
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with right_panel:
    if st.session_state.get('result'):
        res = st.session_state.result
        
        # HITL Alert
        if res.get("needs_human_review"):
            st.warning("⚠️ **Human Review Required**: The system is not fully confident in this solution. Please verify carefully.")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Result Header
        st.markdown(f"""
            <div class="result-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #10b981; font-weight: 600; font-size: 0.75rem;">SOLVED ● CONFIDENCE {int(res['confidence']*100)}%</span>
                    <span style="font-size: 1.2rem;">✅</span>
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #fff; margin-top: 10px;">
                    {res['final_answer']}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<h4 style='margin-top:25px; color:#94a3b8; font-size: 1rem;'>DETAILED EXPLANATION</h4>", unsafe_allow_html=True)
        
        # Scrollable Markdown Area
        st.markdown('<div style="max-height: 500px; overflow-y: auto; padding-right: 15px; background: rgba(0,0,0,0.2); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">', unsafe_allow_html=True)
        if res.get("explanation_md"):
            st.markdown(res["explanation_md"])
        else:
            for i, step in enumerate(res.get("explanation", [])):
                st.markdown(f"**Step {i+1}**: {step}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ Clear & Restart"):
            st.session_state.result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Empty State
        st.markdown(f"""
            <div class="glass-card" style="height: 500px; display: flex; flex-direction: column; justify-content: center; align-items: center; border-style: dashed; border-color: #334155;">
                <div style="background: rgba(99, 102, 241, 0.1); width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                    <span style="font-size: 30px;">🛰️</span>
                </div>
                <h3 style="color: #f8fafc; margin-bottom: 10px;">Waiting for Input</h3>
                <p style="color: #64748b; text-align: center; max-width: 300px;">Enter a mathematical problem to see the agent's logic and final solution.</p>
            </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
    <div style="margin-top: 30px; padding: 20px; text-align: center; border-top: 1px solid rgba(255,255,255,0.05);">
        <p style="color: #475569; font-size: 0.8rem;">Built with LangGraph & Streamlit • Powered by Groq Cloud</p>
    </div>
""", unsafe_allow_html=True)