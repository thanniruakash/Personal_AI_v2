import random

from .loader import KnowledgeLoader
from .search import KnowledgeSearch
from .memory import Memory


class PersonalAI:

    def __init__(self):
        self.memory = Memory()
        self.knowledge = KnowledgeLoader().load()
        self.search = KnowledgeSearch(self.knowledge)

    def ask(self, question):
        question = str(question).strip()

        if not question:
            return "Please type a question."

        # Search local knowledge first.
        results = self.search.search(question)

        if results:
            answer = results[0].get("content", str(results[0]))
        else:
            responses = [
                "I don't know that yet. Please teach me.",
                "I couldn't find that in my knowledge base.",
                "Can you add this topic to my knowledge?"
            ]
            answer = random.choice(responses)

        # Save the conversation in local memory.
        self.memory.add(question, answer)

        return answer
