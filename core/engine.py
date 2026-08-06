import random
from search import KnowledgeSearch
from memory import Memory


class PersonalAI:

    def __init__(self):
        self.search = KnowledgeSearch()
        self.memory = Memory()

    def ask(self, question):

        # Save last question
        self.memory.remember("last_question", question)

        # Search knowledge base
        result = self.search.search(question)

        if result:
            return result

        # Default reply
        responses = [
            "I don't know that yet. Please teach me.",
            "I couldn't find that in my knowledge base.",
            "Can you add this topic to my knowledge?"
        ]

        return random.choice(responses)
