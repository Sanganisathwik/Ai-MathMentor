import re
import json
from typing import Dict, Any, Optional
from backend.llm.groq_client import call_groq

class SolverAgent:
    """
    JEE-level math solver with strict prompt discipline and HITL triggers.
    """

    def solve(self, parsed_problem: Dict, route_plan: Dict, rag_context: Optional[Dict] = None) -> Dict[str, Any]:
        problem_text = parsed_problem.get("problem_text", "No problem provided.")
        topic = parsed_problem.get("topic", "general math")
        
        context_str = ""
        if rag_context and rag_context.get("results"):
            context_str = "\nAvailable Verified Formulas:\n" + "\n".join([f"- {r['content']}" for r in rag_context["results"]])

        prompt = f"""You are a highly accurate JEE-level mathematics tutor.

NON-NEGOTIABLE RULES:
- Do NOT guess.
- Do NOT skip steps.
- Do NOT invent formulas.
- If you are unsure at ANY point, explicitly say so.
- Mathematical correctness is more important than fluency.
- If a mistake is possible, request human verification.
- You must strictly follow the requested output format.

Problem:
{problem_text}
{context_str}

Output FORMAT (DO NOT CHANGE):

TOPIC:
<algebra / calculus / probability / linear algebra>

APPROACH:
<brief plan of how the problem will be solved>

FORMULAS USED:
- Formula 1
- Formula 2

STEP-BY-STEP SOLUTION:
Step 1: ...
Step 2: ...
Step 3: ...

FINAL ANSWER:
<clear final answer>

SELF-CHECK:
- Domain checked: YES / NO
- Any assumptions made: NONE / <list>
- Confidence (0–1): <number>

If Confidence < 0.85, write exactly:
"HUMAN REVIEW REQUIRED"
"""

        llm_output = call_groq(prompt)

        # Robust Parsing
        def extract_section(text, header):
            # Matches header until next uppercase header or end of string
            pattern = rf"{header}:?\s*(.*?)(?=\n[A-Z\s_]+:|(\*\*|#)*[A-Z\s_]+:|(\*\*|#)*\w+\s?[\w\s]*:|(\s*\d+\.\s*)|$)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        parsed_topic = extract_section(llm_output, "TOPIC")
        approach = extract_section(llm_output, "APPROACH")
        formulas = extract_section(llm_output, "FORMULAS USED")
        steps = extract_section(llm_output, "STEP-BY-STEP SOLUTION")
        final_answer = extract_section(llm_output, "FINAL ANSWER")
        self_check = extract_section(llm_output, "SELF-CHECK")

        # Confidence extraction
        confidence = 1.0
        conf_match = re.search(r"Confidence\s*\(0?–1\):\s*([\d.]+)", self_check)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except:
                pass

        needs_hitl = "HUMAN REVIEW REQUIRED" in llm_output or confidence < 0.85

        return {
            "raw_output": llm_output,
            "topic": parsed_topic,
            "approach": approach,
            "formulas": formulas,
            "reasoning": steps,
            "answer": final_answer,
            "confidence": confidence,
            "needs_human_review": needs_hitl,
            "status": "solved" if final_answer else "failed"
        }
