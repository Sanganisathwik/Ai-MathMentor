from fastapi import FastAPI
from backend.api.routes import router


app = FastAPI(
    title="Math Mentor AI Backend",
    description="Multimodal RAG-based Math Tutor",
    version="1.0.0"
)

app.include_router(router)
