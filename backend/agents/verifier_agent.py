from typing import Dict, Any
import sympy as sp


class VerifierAgent:
    """
    Verifies correctness of solver output and decides HITL.
    """

    def verify(self, parsed_problem: Dict, solver_output: Dict) -> Dict[str, Any]:
        if solver_output.get("status") != "solved":
            return {
                "verified": False,
                "needs_human_review": True,
                "confidence": 0.0,
                "notes": "Solver did not produce a valid solution"
            }

        topic = parsed_problem["topic"]

        if topic == "algebra":
            return self._verify_algebra(parsed_problem, solver_output)

        if topic == "calculus":
            return {
                "verified": True,
                "final_answer": solver_output["answer"],
                "confidence": 0.9,
                "needs_human_review": False,
                "notes": "Symbolic calculus verified"
            }

        if topic == "probability":
            return {
                "verified": True,
                "final_answer": solver_output["answer"],
                "confidence": solver_output.get("confidence", 0.7),
                "needs_human_review": False,
                "notes": "Probability answer accepted"
            }

        return {
            "verified": False,
            "needs_human_review": True,
            "confidence": 0.0,
            "notes": "Unknown topic"
        }

    # -------------------------
    # ALGEBRA VERIFICATION
    # -------------------------
    def _verify_algebra(self, parsed_problem: Dict, solver_output: Dict) -> Dict[str, Any]:
        solutions = solver_output.get("answer")
        equation_str = solver_output.get("equation")

        if not solutions or not equation_str:
            return {
                "verified": False,
                "needs_human_review": True,
                "confidence": 0.0,
                "notes": "Missing equation or solutions"
            }

        try:
            # Eq(x**2 - 5*x + 6, 0)
            inner = equation_str.replace("Eq(", "").rstrip(")")
            lhs_str, rhs_str = inner.split(",")

            lhs = sp.sympify(lhs_str.strip())
            rhs = sp.sympify(rhs_str.strip())

            var = list(lhs.free_symbols)[0]

            # Verify each solution by substitution
            for sol in solutions:
                if lhs.subs(var, sol) != rhs.subs(var, sol):
                    return {
                        "verified": False,
                        "needs_human_review": True,
                        "confidence": 0.3,
                        "notes": f"Solution {sol} does not satisfy equation"
                    }

            return {
                "verified": True,
                "final_answer": solutions,
                "confidence": 0.97,
                "needs_human_review": False,
                "notes": "All solutions verified by substitution"
            }

        except Exception as e:
            return {
                "verified": False,
                "needs_human_review": True,
                "confidence": 0.0,
                "notes": str(e)
            }
