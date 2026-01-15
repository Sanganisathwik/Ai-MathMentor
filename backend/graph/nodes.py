from backend.agents.parser_agent import ParserAgent
from backend.agents.router_agent import IntentRouterAgent
from backend.agents.solver_agent import SolverAgent
from backend.agents.verifier_agent import VerifierAgent
from backend.agents.explainer_agent import ExplainerAgent
from backend.rag.orchestrator import multi_rag_retrieve

parser = ParserAgent()
router = IntentRouterAgent()
solver = SolverAgent()
verifier = VerifierAgent()
explainer = ExplainerAgent()

def parse_problem(state):
    parsed = parser.parse(state["raw_input"])
    state["parsed_problem"] = parsed
    state["agent_trace"].append("ParserAgent")
    state["needs_human_review"] = parsed.get("needs_clarification", False)
    return state

def route_intent(state):
    route = router.route(state["parsed_problem"])
    state["route_plan"] = route
    state["agent_trace"].append("IntentRouterAgent")
    return state

def retrieve_rag(state):
    # Only retrieve if plan suggests RAG
    if state["route_plan"].get("use_rag"):
        rag = multi_rag_retrieve(state["parsed_problem"])
        state["rag_context"] = rag
        state["agent_trace"].append("MultiRAG")
    else:
        state["rag_context"] = {"results": []}
    return state

def solve_problem(state):
    solution = solver.solve(
        state["parsed_problem"],
        state["route_plan"],
        state["rag_context"]
    )
    state["solver_output"] = solution
    # Trigger HITL if solver is not confident
    if solution.get("needs_human_review"):
        state["needs_human_review"] = True
        
    state["agent_trace"].append("SolverAgent")
    return state

def verify_solution(state):
    verdict = verifier.verify(
        state["parsed_problem"],
        state["solver_output"]
    )
    state["verifier_output"] = verdict
    # Update needs_human_review based on verification (Persist if already True)
    if verdict.get("needs_human_review"):
        state["needs_human_review"] = True
    state["agent_trace"].append("VerifierAgent")
    return state

def human_review(state):
    # This node is reached when needs_human_review is True
    # In a real LangGraph app, this might involve a breakpoint.
    # For now, we follow the user's logic of updating raw_input if feedback exists.
    if state.get("human_feedback"):
        state["raw_input"] = state["human_feedback"]
        state["needs_human_review"] = False
        state["agent_trace"].append("HITL_Feedback_Applied")
    return state

def explain_solution(state):
    explanation = explainer.explain(
        state["parsed_problem"],
        state["solver_output"],
        state["verifier_output"]
    )
    state["explanation_output"] = explanation
    state["agent_trace"].append("ExplainerAgent")
    return state
