from typing import Dict, Any
import sympy as sp


class SolverAgent:
    """
    Solves math problems using symbolic computation (SymPy).
    """

    def solve(self, parsed_problem: Dict, route_plan: Dict) -> Dict[str, Any]:
        topic = parsed_problem["topic"]
        text = parsed_problem["problem_text"]
        variables = parsed_problem["variables"]

        try:
            if topic == "algebra":
                return self._solve_algebra(text, variables)

            elif topic == "calculus":
                return self._solve_calculus(text, variables)

            elif topic == "probability":
                return self._solve_probability(text)

            else:
                return {
                    "status": "unsupported",
                    "message": f"No solver available for topic: {topic}",
                    "confidence": 0.0
                }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "confidence": 0.0
            }

    # -------------------------
    # ALGEBRA SOLVER
    # -------------------------
    def _solve_algebra(self, text: str, variables: list) -> Dict[str, Any]:
        if "=" not in text:
            return {
                "status": "needs_clarification",
                "message": "Equation missing '='",
                "confidence": 0.0
            }

        lhs, rhs = text.split("=")

        var = sp.symbols(variables[0])
        lhs_expr = sp.sympify(lhs.replace("^", "**"))
        rhs_expr = sp.sympify(rhs.replace("^", "**"))

        equation = sp.Eq(lhs_expr, rhs_expr)
        solution = sp.solve(equation, var)

        return {
            "status": "solved",
            "answer": solution,
            "equation": str(equation),
            "confidence": 0.95
        }

    # -------------------------
    # CALCULUS SOLVER
    # -------------------------
    def _solve_calculus(self, text: str, variables: list) -> Dict[str, Any]:
        var = sp.symbols(variables[0])

        if "derivative" in text:
            expr = self._extract_expression(text)
            derivative = sp.diff(expr, var)

            return {
                "status": "solved",
                "answer": str(derivative),
                "operation": "derivative",
                "confidence": 0.9
            }

        if "limit" in text and "->" in text:
            expr, limit_value = self._extract_limit(text)
            limit_result = sp.limit(expr, var, limit_value)

            return {
                "status": "solved",
                "answer": str(limit_result),
                "operation": "limit",
                "confidence": 0.9
            }

        return {
            "status": "needs_clarification",
            "message": "Unsupported calculus operation",
            "confidence": 0.0
        }

    # -------------------------
    # PROBABILITY SOLVER
    # -------------------------
    def _solve_probability(self, text: str) -> Dict[str, Any]:
        if "coin" in text and "head" in text:
            return {
                "status": "solved",
                "answer": "1/2",
                "explanation": "A fair coin has equal probability for head and tail.",
                "confidence": 0.8
            }

        return {
            "status": "needs_clarification",
            "message": "Probability problem too generic",
            "confidence": 0.3
        }

    # -------------------------
    # HELPERS
    # -------------------------
    def _extract_expression(self, text: str):
        expr_text = text.split("of")[-1]
        expr_text = expr_text.replace("^", "**")
        return sp.sympify(expr_text)

    def _extract_limit(self, text: str):
        parts = text.split("->")
        limit_value = float(parts[-1].strip())
        expr = sp.symbols("x")
        return expr, limit_value
