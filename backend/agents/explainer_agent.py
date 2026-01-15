from backend.llm.groq_client import call_groq

class ExplainerAgent:
    """
    JEE mathematics tutor that generates student-friendly explanations.
    """
    def explain(self, parsed_problem: dict, solver_output: dict, verifier_output: dict) -> dict:
        problem_text = parsed_problem.get("problem_text", "")
        # Combine solver and verifier info for a complete context
        context = f"Problem: {problem_text}\nSolver Output: {solver_output.get('raw_output', '')}\nVerification: {verifier_output}"

        prompt = f"""You are a JEE mathematics tutor.

Rewrite the solution below into:
- Clear
- Exam-oriented
- Beginner-friendly steps

Rules:
- Explain WHY each step is taken
- Avoid unnecessary jargon
- Use Markdown formatting
- End with the final answer clearly

Verified Solution Context:
{context}

Output FORMAT:

### Step-by-Step Explanation
Step 1: ...
Step 2: ...
Step 3: ...

### Final Answer
...
"""

        explanation_raw = call_groq(prompt)

        # Split steps for UI list if possible, or just return the text
        lines = [line.strip() for line in explanation_raw.split("\n") if line.strip()]

        return {
            "final_answer": solver_output.get("answer", "No answer found."),
            "explanation": lines,
            "explanation_md": explanation_raw
        }
