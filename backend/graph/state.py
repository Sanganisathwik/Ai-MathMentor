from typing import TypedDict, List, Optional, Dict, Any

class MathMentorState(TypedDict):
    # Input
    raw_input: str
    input_type: str  # text | image | audio

    # Parsed
    parsed_problem: Optional[Dict[str, Any]]

    # Routing
    route_plan: Optional[Dict[str, Any]]

    # RAG
    rag_context: Optional[Dict[str, Any]]

    # Solver
    solver_output: Optional[Dict[str, Any]]

    # Verifier
    verifier_output: Optional[Dict[str, Any]]

    # Explanation
    explanation_output: Optional[Dict[str, Any]]

    # HITL
    needs_human_review: bool
    human_feedback: Optional[str]

    # Trace
    agent_trace: List[str]
