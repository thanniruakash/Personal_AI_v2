import os
import json

all_intents = []

intent_folder = "intents"

for filename in os.listdir(intent_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(intent_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_intents.extend(data["intents"])

print(f"Loaded {len(all_intents)} intents.")

for intent in all_intents:
    print(intent["tag"])
