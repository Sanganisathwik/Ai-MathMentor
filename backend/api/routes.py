from fastapi import APIRouter, UploadFile, File
import shutil
import os
from backend.tools.ocr import extract_text_from_image
from backend.tools.asr import transcribe_audio
from backend.api.schemas import TextInput, SolutionResponse
from backend.memory.memory_store import MemoryStore
from backend.memory.similarity_search import find_similar
from backend.rag.retriever import retrieve

from backend.agents.parser_agent import ParserAgent
from backend.agents.router_agent import IntentRouterAgent
from backend.agents.solver_agent import SolverAgent
from backend.agents.verifier_agent import VerifierAgent
from backend.agents.explainer_agent import ExplainerAgent

router = APIRouter()

@router.get("/health/asr")
def health_asr():
    import shutil
    import os
    from backend.tools.asr import model
    matches = []
    if os.name == "nt":
        winget_path = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
        if os.path.exists(winget_path):
            import glob
            matches = glob.glob(os.path.join(winget_path, "Gyan.FFmpeg*", "**", "bin"), recursive=True)

    return {
        "ffmpeg_found": shutil.which("ffmpeg") is not None,
        "ffmpeg_path": shutil.which("ffmpeg"),
        "model_loaded": model is not None,
        "path": os.environ.get("PATH"),
        "winget_matches": matches
    }

# Initialize memory
memory_store = MemoryStore()


from backend.graph.graph import graph

@router.post("/solve", response_model=SolutionResponse)
def solve_problem(input_data: TextInput):
    # Initial state for the graph
    initial_state = {
        "raw_input": input_data.text,
        "input_type": "text",
        "parsed_problem": None,
        "route_plan": None,
        "rag_context": None,
        "solver_output": None,
        "verifier_output": None,
        "explanation_output": None,
        "needs_human_review": False,
        "human_feedback": None,
        "agent_trace": []
    }

    # Configuration for MemorySaver (checkpointer)
    # In a real app, thread_id would come from the session or user_id
    config = {"configurable": {"thread_id": "default_user"}}

    # Execute the graph
    final_output = graph.invoke(initial_state, config=config)

    # Map graph state back to the expected API response
    explanation = final_output.get("explanation_output", {})
    verifier = final_output.get("verifier_output", {})
    rag = final_output.get("rag_context", {})
    
    # Format retrieved context for the UI
    ui_context = []
    if rag and rag.get("results"):
        ui_context = [
            {
                "source": r["source"],
                "content": r["content"],
                "score": float(r["score"])
            }
            for r in rag["results"]
        ]

    return {
        "final_answer": explanation.get("final_answer"),
        "explanation": explanation.get("explanation", []),
        "explanation_md": explanation.get("explanation_md"),
        "confidence": verifier.get("confidence", 0.0),
        "retrieved_context": ui_context,
        "agent_trace": final_output.get("agent_trace", []),
        "needs_human_review": final_output.get("needs_human_review", False)
    }


@router.post("/feedback")
def submit_feedback(feedback: dict):
    memory = memory_store.load()
    if not memory:
        return {"status": "no memory to update"}

    memory[-1]["feedback"] = feedback
    import json
    with open("backend/data/memory_logs/memory.json", "w") as f:
        json.dump(memory, f, indent=2)

    return {"status": "feedback recorded"}
@router.post("/solve/image")
def solve_from_image(file: UploadFile = File(...)):
    os.makedirs("backend/data/uploads", exist_ok=True)
    image_path = f"backend/data/uploads/{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ocr_result = extract_text_from_image(image_path)

    return {
        "extracted_text": ocr_result.get("extracted_text", ""),
        "confidence": ocr_result.get("confidence", 0.0)
    }


@router.post("/solve/audio")
def solve_from_audio(file: UploadFile = File(...)):
    """
    Solve math problem from audio input (Sync endpoint to prevent event loop blocking)
    """
    audio_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.file.read())
            audio_path = tmp.name

        # Transcribe
        asr_result = transcribe_audio(audio_path)
        
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        if asr_result.get("error"):
            # If model found an error but didn't crash
            raise HTTPException(status_code=500, detail=asr_result["error"])

        return {
            "extracted_text": asr_result.get("extracted_text", ""),
            "confidence": asr_result.get("confidence", 0.0)
        }

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print("🔥 [AUDIO ERROR]:", err_msg)
        
        # Cleanup on failure
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass
            
        raise HTTPException(status_code=500, detail=str(e))
