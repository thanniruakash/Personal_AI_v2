import json
import os
import random
from datetime import datetime

from knowledge_loader import KnowledgeLoader
from search import KnowledgeSearch


class PersonalAI:

    def __init__(self):
        self.loader = KnowledgeLoader()
        self.search = KnowledgeSearch()
        
    def search_intents(self, message):

        message_words = set(message.lower().split())

        intents_folder = "intents"

        best_response = None
        best_score = 0

        for filename in os.listdir(intents_folder):

            if filename.endswith(".json"):

                filepath = os.path.join(intents_folder, filename)

                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for intent in data["intents"]:

                    for pattern in intent["patterns"]:

                        pattern_words = set(pattern.lower().split())

                        score = len(message_words & pattern_words)

                        if score > best_score:
                            best_score = score
                            best_response = random.choice(intent["responses"])

        return best_response

    def ask(self, message):

        msg = message.lower().strip()

        # Greetings
        if msg in ["hi", "hello", "hey"]:
            return "Hello! I'm Personal AI. How can I help you today?"

        # Goodbye
        if msg in ["bye", "goodbye", "exit", "quit"]:
            return "Goodbye! Have a great day."

        # Offline Date & Time
        if "date" in msg:
            return datetime.now().strftime("Today's date is %d %B %Y.")

        if "time" in msg:
            return datetime.now().strftime("Current time is %I:%M %p.")

        if "day" in msg:
            return datetime.now().strftime("Today is %A.")

        if "month" in msg:
            return datetime.now().strftime("Current month is %B.")

        if "year" in msg:
            return datetime.now().strftime("Current year is %Y.")

        # Search knowledge files first
        response = self.search.search(message)

        if response:
            return response

        # Search intents if not found
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

        if message.lower() in ["quit", "exit", "bye"]:
            print("AI: Goodbye! Have a great day.")
            break

        print("AI:", ai.ask(message))
