from fastapi import APIRouter
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

# Initialize agents (singleton style)
parser_agent = ParserAgent()
router_agent = IntentRouterAgent()
solver_agent = SolverAgent()
verifier_agent = VerifierAgent()
explainer_agent = ExplainerAgent()

# Initialize memory
memory_store = MemoryStore()


@router.post("/solve", response_model=SolutionResponse)
def solve_problem(input_data: TextInput):
    agent_trace = []
    retrieved_context = []

    # 1. PARSER
    parsed_problem = parser_agent.parse(input_data.text)
    agent_trace.append("ParserAgent")

    if parsed_problem["needs_clarification"]:
        return {
            "final_answer": None,
            "explanation": ["Problem needs clarification before solving."],
            "confidence": 0.0,
            "retrieved_context": [],
            "agent_trace": agent_trace,
            "needs_human_review": True
        }

    # 2. ROUTER
    route_plan = router_agent.route(parsed_problem)
    agent_trace.append("IntentRouterAgent")

    # 3. MEMORY LOOKUP (optional reuse signal)
    memory_records = memory_store.load()
    similar_cases = find_similar(memory_records, parsed_problem["problem_text"])
    if similar_cases:
        past, score = similar_cases[0]
        if past.get("verifier", {}).get("verified"):
            agent_trace.append("MemoryReuse")

    # 4. RAG RETRIEVAL
   # 4. RAG RETRIEVAL
if route_plan.get("use_rag"):
    raw_context = retrieve(parsed_problem["problem_text"])
    retrieved_context = [
        {
            "source": c["source"],
            "content": c["content"],
            "score": float(c["score"])
        }
        for c in raw_context
    ]


    # 5. SOLVER
    solver_output = solver_agent.solve(parsed_problem, route_plan)
    agent_trace.append("SolverAgent")

    # 6. VERIFIER
    verifier_output = verifier_agent.verify(parsed_problem, solver_output)
    agent_trace.append("VerifierAgent")

    if verifier_output["needs_human_review"]:
        return {
            "final_answer": None,
            "explanation": ["Solution could not be verified with confidence."],
            "confidence": verifier_output.get("confidence", 0.0),
            "retrieved_context": retrieved_context,
            "agent_trace": agent_trace,
            "needs_human_review": True
        }

    # 7. EXPLAINER
    explanation_output = explainer_agent.explain(
        parsed_problem,
        solver_output,
        verifier_output
    )
    agent_trace.append("ExplainerAgent")

    # 8. SAVE TO MEMORY
    memory_store.save({
        "input_text": input_data.text,
        "parsed_problem": parsed_problem,
        "retrieved_context": retrieved_context,
        "solver": solver_output,
        "verifier": verifier_output,
        "explainer": explanation_output
    })

    return {
        "final_answer": explanation_output["final_answer"],
        "explanation": explanation_output["explanation"],
        "confidence": verifier_output.get("confidence", 0.9),
        "retrieved_context": retrieved_context,
        "agent_trace": agent_trace,
        "needs_human_review": False
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
