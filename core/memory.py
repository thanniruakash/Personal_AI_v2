class KnowledgeMemory:
    def __init__(self):
        self.history = []

    def add(self, question, answer):
        self.history.append({
            "question": question,
            "answer": answer
        })

    def remember(self, key, value):
        """Store a simple key/value memory entry."""
        self.history.append({
            key: value
        })

    def last(self, n=5):
        return self.history[-n:]


# Backward-compatible name used by the AI engine.
Memory = KnowledgeMemory
