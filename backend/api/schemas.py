from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# -------------------------
# INPUT MODELS
# -------------------------

class TextInput(BaseModel):
    text: str


class ImageInput(BaseModel):
    image_path: str          # stored path after upload
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None


class AudioInput(BaseModel):
    audio_path: str          # stored path after upload
    transcript: Optional[str] = None
    asr_confidence: Optional[float] = None


# -------------------------
# PARSER OUTPUT
# -------------------------

class ParsedProblem(BaseModel):
    problem_text: str
    topic: str
    variables: List[str]
    constraints: List[str]
    needs_clarification: bool


# -------------------------
# RAG CONTEXT
# -------------------------

class RetrievedChunk(BaseModel):
    source: str
    content: str
    score: float


# -------------------------
# SOLUTION OUTPUT
# -------------------------

from pydantic import BaseModel
from typing import List, Optional


class RetrievedChunk(BaseModel):
    source: str
    content: str
    score: float


class SolutionResponse(BaseModel):
    final_answer: Optional[str]
    explanation: List[str]
    confidence: float
    retrieved_context: List[RetrievedChunk]
    agent_trace: List[str]
    needs_human_review: bool



# -------------------------
# FEEDBACK / HITL
# -------------------------

class FeedbackInput(BaseModel):
    problem_id: str
    is_correct: bool
    corrected_solution: Optional[str] = None
    comments: Optional[str] = None


# -------------------------
# MEMORY SEARCH
# -------------------------

class MemorySearchInput(BaseModel):
    query: str
    top_k: int = 3
from pydantic import BaseModel
from typing import List


class RetrievedChunk(BaseModel):
    source: str
    content: str
    score: float


