import json
import os
import torch
import numpy as np

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

all_words = []
tags = []
xy = []

intents_folder = "intents"

for filename in os.listdir(intents_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(intents_folder, filename)

        with open(filepath, "r") as f:
            intents = json.load(f)

        for intent in intents["intents"]:
            tag = intent["tag"]
            tags.append(tag)

            for pattern in intent["patterns"]:
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w, tag))

ignore_words = ['?', '.', '!', ',']

all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

print(f"Loaded {len(tags)} tags")
print(f"Loaded {len(all_words)} words")
print("Training data prepared successfully.")
