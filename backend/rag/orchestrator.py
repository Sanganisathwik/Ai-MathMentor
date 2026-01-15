from backend.rag.retriever import retrieve

def multi_rag_retrieve(parsed_problem):
    """
    Orchestrates retrieval from multiple sources if needed.
    Currently wraps the standard TF-IDF retriever.
    """
    problem_text = parsed_problem.get("problem_text", "")
    if not problem_text:
        return {"results": []}
        
    results = retrieve(problem_text)
    return {
        "results": results,
        "source_count": len(results)
    }
