import os
import shutil
import platform
import glob

# --- FFmpeg Path Fix for Windows (Auto-Detection) ---
def _inject_ffmpeg():
    if platform.system() == "Windows":
        if not shutil.which("ffmpeg"):
            # 1. Check known Winget path from previous discovery
            winget_known = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_859a1928a55c5\ffmpeg-8.0.1-full_build\bin")
            
            # 2. Dynamic search as fallback
            winget_root = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
            found_path = None
            
            if os.path.exists(winget_known):
                found_path = winget_known
            elif os.path.exists(winget_root):
                matches = glob.glob(os.path.join(winget_root, "Gyan.FFmpeg*", "**", "ffmpeg.exe"), recursive=True)
                if matches:
                    found_path = os.path.dirname(matches[0])
            
            if found_path:
                os.environ["PATH"] = found_path + os.pathsep + os.environ.get("PATH", "")
                print(f"--- [ASR] FFmpeg path detected and injected: {found_path} ---")
            else:
                print("--- [ASR] FFmpeg binary not found in WinGet packages ---")

_inject_ffmpeg()

import whisper
# Load model once at module level
model = whisper.load_model("base")

def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio file to text using Whisper
    """
    try:
        if not os.path.exists(audio_path):
            return {"extracted_text": "", "confidence": 0.0, "error": "File not found"}
        
        # Final sanity check for FFmpeg
        ff_path = shutil.which("ffmpeg")
        print(f"--- [ASR] transcribe_audio: shutil.which('ffmpeg') = {ff_path} ---")
        if not ff_path:
             print(f"--- [ASR] Current PATH: {os.environ.get('PATH')} ---")
            
        result = model.transcribe(audio_path)

        text = result.get("text", "").strip()
        confidence = 0.9 if len(text) > 10 else 0.4

        return {
            "extracted_text": text,
            "confidence": confidence
        }
    except Exception as e:
        return {
            "extracted_text": "",
            "confidence": 0.0,
            "error": str(e)
        }
