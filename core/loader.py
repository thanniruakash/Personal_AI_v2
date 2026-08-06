import os
import json

class KnowledgeLoader:
    def __init__(self):
        self.base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "knowledge"
        )

    def load(self):
        knowledge = []

        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                            if isinstance(data, list):
                                knowledge.extend(data)
                            elif isinstance(data, dict):
                                knowledge.append(data)

                    except Exception:
                        pass

        return knowledge
