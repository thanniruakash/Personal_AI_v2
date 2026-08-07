import json
import os
import random
from datetime import datetime

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

                        if (
                            pattern.lower() in message.lower()
                            or message.lower() in pattern.lower()
                        ):
                            return random.choice(intent["responses"])

        return None

    def ask(self, message):

        msg = message.lower()

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
