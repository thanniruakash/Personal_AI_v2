import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Personal AI"

# Load all intent files
all_intents = []

import os

for filename in os.listdir("intents"):
    if filename.endswith(".json"):
        with open(os.path.join("intents", filename), "r", encoding="utf-8") as f:
            data = json.load(f)

            if "intents" in data:
                all_intents.extend(data["intents"])

print("=" * 50)
print("🤖 Personal AI v2")
print("=" * 50)
print("Type 'quit' to exit.\n")

while True:

    sentence = input("You: ")

    if sentence.lower() == "quit":
        break

    sentence_words = tokenize(sentence)
    X = bag_of_words(sentence_words, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    found = False

    for intent in all_intents:

        if tag == intent["tag"]:

            print(f"{bot_name}: {random.choice(intent['responses'])}")
            found = True
            break

    if not found:
        print(f"{bot_name}: Sorry, I don't know that yet.")
