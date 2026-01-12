import json
import os
from datetime import datetime

MEMORY_PATH = "backend/data/memory_logs/memory.json"


class MemoryStore:
    def __init__(self):
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        if not os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, "w") as f:
                json.dump([], f)

    def load(self):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)

    def save(self, record: dict):
        memory = self.load()
        record["timestamp"] = datetime.utcnow().isoformat()
        memory.append(record)
        with open(MEMORY_PATH, "w") as f:
            json.dump(memory, f, indent=2)
