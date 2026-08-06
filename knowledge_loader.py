import os

class KnowledgeLoader:
    def __init__(self, knowledge_path="knowledge"):
        self.knowledge_path = knowledge_path

    def get_topics(self):
        topics = []

        if not os.path.exists(self.knowledge_path):
            return topics

        for folder in os.listdir(self.knowledge_path):
            folder_path = os.path.join(self.knowledge_path, folder)

            if os.path.isdir(folder_path):
                topics.append(folder)

        return topics

    def list_files(self, topic):
        folder = os.path.join(self.knowledge_path, topic)

        if not os.path.exists(folder):
            return []

        return os.listdir(folder)
