import json
import torch
import numpy as np

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# Load intents
with open("intents/intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Process intents
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!", ","]
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

print(f"Total words: {len(all_words)}")
print(f"Total tags: {len(tags)}")

print("Training data prepared successfully.")
