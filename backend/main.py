from fastapi import FastAPI
from backend.api.routes import router


app = FastAPI(
    title="Math Mentor AI Backend",
    description="Multimodal RAG-based Math Tutor",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"status": "alive", "service": "Math Mentor AI Backend"}

@app.get("/health/asr")
def health_asr():
    import shutil
    from backend.tools.asr import model
    return {
        "ffmpeg_found": shutil.which("ffmpeg") is not None,
        "ffmpeg_path": shutil.which("ffmpeg"),
        "model_loaded": model is not None,
        "path": os.environ.get("PATH")
    }
