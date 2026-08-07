import json
import os
import random

from knowledge_loader import KnowledgeLoader


class PersonalAI:

    def __init__(self):
        self.loader = KnowledgeLoader()

    def search_intents(self, message):

        intents_folder = "intents"

        for filename in os.listdir(intents_folder):

            if filename.endswith(".json"):

                filepath = os.path.join(intents_folder, filename)

                with open(filepath, "r", encoding="utf-8") as f:

                    data = json.load(f)

                for intent in data["intents"]:

                    for pattern in intent["patterns"]:

                        if pattern.lower() == message.lower():

                            return random.choice(intent["responses"])

        return None

    def ask(self, message):

        response = self.search_intents(message)

        if response:
            return response

        return (
            "I don't know the answer yet. "
            "Please add it to my knowledge base."
        )
if __name__ == "__main__":
    ai = PersonalAI()

    print("=== Personal AI v2 ===")
    print("Type 'quit' to exit.\n")

    while True:
        message = input("You: ")

        if message.lower() == "quit":
            print("Goodbye!")
            break

        print("AI:", ai.ask(message))
