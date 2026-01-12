def apply_feedback(memory_record: dict, is_correct: bool, comment: str | None):
    memory_record["feedback"] = {
        "is_correct": is_correct,
        "comment": comment
    }
    return memory_record
