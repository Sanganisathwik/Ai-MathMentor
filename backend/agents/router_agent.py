from typing import Dict


class IntentRouterAgent:
    """
    Routes the parsed problem to the correct solving strategy.
    """

    def route(self, parsed_problem: Dict) -> Dict:
        topic = parsed_problem.get("topic", "algebra")

        # Default routing plan
        plan = {
            "route": "algebra_solver",
            "tools": [],
            "use_rag": True,
            "needs_calculator": False
        }

        if topic == "algebra":
            plan.update({
                "route": "algebra_solver",
                "tools": ["sympy"],
                "needs_calculator": True
            })

        elif topic == "probability":
            plan.update({
                "route": "probability_solver",
                "tools": [],
                "needs_calculator": False
            })

        elif topic == "calculus":
            plan.update({
                "route": "calculus_solver",
                "tools": ["sympy"],
                "needs_calculator": True
            })

        elif topic == "linear_algebra":
            plan.update({
                "route": "linear_algebra_solver",
                "tools": ["numpy"],
                "needs_calculator": True
            })

        return plan
