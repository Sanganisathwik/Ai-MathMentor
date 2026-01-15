import json
from backend.llm.groq_client import call_groq

class VerifierAgent:
    """
    Strict JEE mathematics verifier that returns structured analysis.
    """
    def verify(self, parsed_problem: dict, solver_output: dict) -> dict:
        problem_text = parsed_problem.get("problem_text", "")
        raw_solver_output = solver_output.get("raw_output", "")

        prompt = f"""You are a strict JEE mathematics verifier.

Your job is to FIND ERRORS.

Check:
- Formula correctness
- Logical flow
- Domain restrictions
- Final answer validity

Original Problem:
{problem_text}

Solver Output:
{raw_solver_output}

Output ONLY JSON (NO EXTRA TEXT):

{{
  "verified": true | false,
  "confidence": 0.0-1.0,
  "errors_found": ["..."],
  "needs_human_review": true | false
}}

RULES:
- If ANY step is unjustified → verified = false
- If confidence < 0.9 → needs_human_review = true
"""

        response = call_groq(prompt)
        
        try:
            # Clean response of any markdown code blocks if necessary
            clean_json = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            
            # Ensure required keys exist
            if "verified" not in result: result["verified"] = False
            if "confidence" not in result: result["confidence"] = 0.5
            if "needs_human_review" not in result: result["needs_human_review"] = True
            
            return result
        except Exception as e:
            # Fallback for parsing error
            return {
                "verified": False,
                "confidence": 0.0,
                "errors_found": [f"Verification Parsing Error: {str(e)}", f"Raw response: {response[:100]}"],
                "needs_human_review": True
            }
