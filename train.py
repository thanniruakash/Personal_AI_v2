import os
import json
import torch
import numpy as np

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

all_words = []
tags = []
xy = []

intent_folder = "intents"

for filename in os.listdir(intent_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(intent_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for intent in data["intents"]:
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
