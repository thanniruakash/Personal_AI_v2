import os
import json


class KnowledgeLoader:

    def __init__(self):
        self.knowledge = {}
        self.load_knowledge()

    def load_knowledge(self):

        base_folder = "knowledge"

        if not os.path.exists(base_folder):
            return

        for subject in os.listdir(base_folder):

            subject_path = os.path.join(base_folder, subject)

            if not os.path.isdir(subject_path):
                continue

            self.knowledge[subject] = []

            for file in os.listdir(subject_path):

                if file.endswith(".json"):

                    filepath = os.path.join(subject_path, file)

                    try:

                        with open(filepath, "r", encoding="utf-8") as f:

                            data = json.load(f)

                            self.knowledge[subject].append(data)

                    except Exception as e:

                        print(f"Error loading {filepath}: {e}")

    def get_subjects(self):
        return list(self.knowledge.keys())

    def get_subject(self, subject):
        return self.knowledge.get(subject.lower(), [])
