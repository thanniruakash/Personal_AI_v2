class KnowledgeMemory:
    def __init__(self):
        self.history = []

    def add(self, question, answer):
        self.history.append({"question": question, "answer": answer})

    def last(self, n=5):
        return self.history[-n:]
