import json
import os


class KnowledgeSearch:

    def __init__(self):
        self.folder = "knowledge"

    def search(self, query):
        query = query.lower()

        for root, dirs, files in os.walk(self.folder):
            for file in files:
                if not file.endswith(".json"):
                    continue

                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                # If JSON contains a list
                if isinstance(data, list):
                    for item in data:
                        result = self.find_answer(item, query)
                        if result:
                            return result

                # If JSON contains a dictionary
                elif isinstance(data, dict):
                    result = self.find_answer(data, query)
                    if result:
                        return result

        return None

    def find_answer(self, item, query):

        text = json.dumps(item).lower()

        if query not in text:
            return None

        if "syntax" in query and "syntax" in item:
            return item["syntax"]

        if "example" in query and "example" in item:
            return item["example"]

        if "code" in query and "code" in item:
            return item["code"]

        if "definition" in query and "definition" in item:
            return item["definition"]

        if "explain" in query and "explanation" in item:
            return item["explanation"]

        if "description" in item:
            return item["description"]

        if "definition" in item:
            return item["definition"]

        return None
