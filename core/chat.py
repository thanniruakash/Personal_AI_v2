import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("🤖 Personal AI v2")
print("=" * 50)

print("Type 'quit' to exit.\n")

while True:
    sentence = input("You: ")

    if sentence.lower() == "quit":
        break

    print("AI: I'm still under development. 😊")
