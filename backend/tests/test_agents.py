from backend.agents.verifier_agent import VerifierAgent


def test_verifier():
    verifier = VerifierAgent()

    parsed_problem = {
        "problem_text": "x^2 - 5*x + 6 = 0",
        "topic": "algebra",
        "variables": ["x"],
        "constraints": [],
        "needs_clarification": False
    }

    solver_output = {
        "status": "solved",
        "answer": [2, 3],
        "equation": "Eq(x**2 - 5*x + 6, 0)",
        "confidence": 0.95
    }

    result = verifier.verify(parsed_problem, solver_output)

    print("\nVerifier Output:")
    print(result)


test_verifier()
from backend.agents.explainer_agent import ExplainerAgent


def test_explainer():
    explainer = ExplainerAgent()

    parsed_problem = {
        "problem_text": "x^2 - 5*x + 6 = 0",
        "topic": "algebra",
        "variables": ["x"],
        "constraints": [],
        "needs_clarification": False
    }

    solver_output = {
        "status": "solved",
        "answer": [2, 3],
        "equation": "Eq(x**2 - 5*x + 6, 0)",
        "confidence": 0.95
    }

    verifier_output = {
        "verified": True,
        "final_answer": [2, 3],
        "confidence": 0.97,
        "needs_human_review": False
    }

    result = explainer.explain(parsed_problem, solver_output, verifier_output)

    print("\nExplainer Output:")
    print(result)


test_explainer()
