import requests

BACKEND_URL = "http://127.0.0.1:8000"


def solve_problem(text: str):
    response = requests.post(
        f"{BACKEND_URL}/solve",
        json={"text": text},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def send_feedback(is_correct: bool, comment: str | None = None):
    return requests.post(
        f"{BACKEND_URL}/feedback",
        json={
            "is_correct": is_correct,
            "comment": comment
        }
    )
