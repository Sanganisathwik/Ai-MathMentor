from typing import Dict, Any, List


class ExplainerAgent:
    """
    Converts verified solutions into step-by-step explanations.
    """

    def explain(
        self,
        parsed_problem: Dict,
        solver_output: Dict,
        verifier_output: Dict
    ) -> Dict[str, Any]:

        if not verifier_output.get("verified"):
            return {
                "final_answer": None,
                "explanation": [
                    "The solution could not be verified with confidence.",
                    "A human review is required before explanation."
                ]
            }

        topic = parsed_problem["topic"]

        if topic == "algebra":
            return self._explain_algebra(parsed_problem, solver_output)

        if topic == "calculus":
            return self._explain_calculus(parsed_problem, solver_output)

        if topic == "probability":
            return self._explain_probability(parsed_problem, solver_output)

        return {
            "final_answer": solver_output.get("answer"),
            "explanation": ["Explanation not available for this topic."]
        }

    # -------------------------
    # ALGEBRA EXPLANATION
    # -------------------------
    def _explain_algebra(self, parsed_problem: Dict, solver_output: Dict) -> Dict[str, Any]:
        equation = solver_output.get("equation")
        solutions = solver_output.get("answer")

        steps: List[str] = []

        steps.append(f"Given equation: {equation.replace('Eq', '').strip()}")
        steps.append("This is a quadratic equation.")
        steps.append("We factorize the quadratic expression.")
        steps.append("Set each factor equal to zero.")

        for sol in solutions:
            steps.append(f"Solving gives x = {sol}")

        final_answer = "x = " + ", ".join(str(sol) for sol in solutions)

        return {
            "final_answer": final_answer,
            "explanation": steps
        }

    # -------------------------
    # CALCULUS EXPLANATION
    # -------------------------
    def _explain_calculus(self, parsed_problem: Dict, solver_output: Dict) -> Dict[str, Any]:
        operation = solver_output.get("operation")
        answer = solver_output.get("answer")

        steps = [
            f"The problem involves finding a {operation}.",
            "We apply standard calculus rules.",
            f"The resulting expression is {answer}."
        ]

        return {
            "final_answer": answer,
            "explanation": steps
        }

    # -------------------------
    # PROBABILITY EXPLANATION
    # -------------------------
    def _explain_probability(self, parsed_problem: Dict, solver_output: Dict) -> Dict[str, Any]:
        answer = solver_output.get("answer")

        steps = [
            "A fair coin has two equally likely outcomes: Head and Tail.",
            "The probability of getting a head is the number of favorable outcomes divided by total outcomes.",
            "Therefore, the probability is 1/2."
        ]

        return {
            "final_answer": answer,
            "explanation": steps
        }
