from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from backend.graph.state import MathMentorState
from backend.graph.nodes import (
    parse_problem,
    route_intent,
    retrieve_rag,
    solve_problem,
    verify_solution,
    human_review,
    explain_solution
)

# Initialize the state graph
builder = StateGraph(MathMentorState)

# Add nodes
builder.add_node("parse", parse_problem)
builder.add_node("route", route_intent)
builder.add_node("rag", retrieve_rag)
builder.add_node("solve", solve_problem)
builder.add_node("verify", verify_solution)
builder.add_node("human_review", human_review)
builder.add_node("explain", explain_solution)

# Set entry point
builder.set_entry_point("parse")

# Standard edges
builder.add_edge("parse", "route")
builder.add_edge("route", "rag")
builder.add_edge("rag", "solve")
builder.add_edge("solve", "verify")

# Conditional edges for HITL / Verification
builder.add_conditional_edges(
    "verify",
    # Decide whether to go to human_review or explain based on the state
    lambda s: "human_review" if s["needs_human_review"] else "explain",
)

# Branch for human intervention - only loop back if feedback is provided
builder.add_conditional_edges(
    "human_review",
    lambda s: "parse" if s.get("human_feedback") else "explain",
)

# Terminal edge
builder.add_edge("explain", "__end__")

# Compile with memory for state persistence
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
