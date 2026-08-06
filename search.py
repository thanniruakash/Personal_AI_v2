import json
import os


class KnowledgeSearch:

    def __init__(self):
        self.folder = "knowledge"

    def search(self, query):

        query = query.lower()

        for root, dirs, files in os.walk(self.folder):

            for file in files:

                if file.endswith(".json"):

                    path = os.path.join(root, file)

                    with open(path, "r", encoding="utf-8") as f:

                        try:
                            data = json.load(f)
                        except:
                            continue

                    text = json.dumps(data).lower()

                    if query in text:
                        return data

        return None
