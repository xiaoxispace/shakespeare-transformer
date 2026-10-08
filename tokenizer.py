"""
This file takes output.txt, the concatenated play corpus, and then tokenize it. 
The tokenizer groups consecutive characters of the same type (alpha, digit, or else).
Executing this file produces 
  1. tokenizer mapping in json, which is a 1-to-1 mapping from string to integer and
  2. token index of the corpus stored in pytorch tensor of type torch.int32
  3. a json string-to-integer map that denotes each token's number of occurrences. 
"""

import logging
import json
import torch
import random
import matplotlib.pyplot as plt

from collections import Counter
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tokenizer")


def load_corpus(corpus_path):
  with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = f.read()
  return corpus


def build_vocabulary(tokens):
  """Build token-to-index and token-count dictionaries."""
  token_counts = Counter(tokens)

  # Sort for deterministic vocabulary construction.
  vocab = sorted(token_counts, reverse=True)

  vocab_to_ind = {
    token: i
    for i, token in enumerate(vocab)
  }

  return vocab_to_ind, dict(token_counts)


def build_token_tensor(tokens, vocab_to_ind):
  token_ids = [vocab_to_ind[t] for t in tokens]
  token_dataset = torch.tensor(token_ids, dtype=torch.int32)
  return token_dataset


class WordTokenizer():
  def __init__(self):
    pass

  def tokenize_play(self, play_string):
    """Tokenize text into words, individual digits, whitespace, and punctuation."""
    tokens = []

    i = 0
    while i < len(play_string):
      c = play_string[i]

      # Group consecutive letters into one token.
      if c.isalpha():
        j = i + 1
        while j < len(play_string) and play_string[j].isalpha():
          j += 1
        tokens.append(play_string[i:j])
        i = j
      # Each digit is its own token.
      elif c.isdigit():
        tokens.append(c)
        i += 1
      # Everything else is one token.
      else:
        tokens.append(c)
        i += 1

    return tokens


class CharTokenizer():
  def __init__(self):
    pass

  def tokenize_play(self, play_string):
    """Tokenize text at character level."""
    tokens = []
    i = 0
    while i < len(play_string):
      c = play_string[i]
      tokens.append(c)
      i += 1
    return tokens


def plot_token_counts(token_counts, num_random=15):
  # -------------------------
  # 1. Top 15 most frequent words
  # -------------------------
  tokens = sorted(
    token_counts.items(),
    key=lambda x: x[1],
    reverse=True
  )[:15]
  counts = [count for _, count in tokens]
  tokens = [token for token, _ in tokens]

  plt.figure(figsize=(12, 5))
  plt.bar(tokens, counts)
  plt.xlabel("Token")
  plt.ylabel("Count")
  plt.title("Top 15 Most Frequent Tokens")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  plt.savefig("figs/top_15_tokens.png", dpi=300)
  plt.close()

  # -------------------------
  # 2. Randomly sampled tokens
  # -------------------------
  num_random = min(num_random, len(token_counts))

  sampled_tokens = random.sample(
    list(token_counts.keys()),
    num_random
  )

  sampled_counts = [
    token_counts[token]
    for token in sampled_tokens
  ]

  plt.figure(figsize=(12, 5))
  plt.bar(sampled_tokens, sampled_counts)
  plt.xlabel("Token")
  plt.ylabel("Count")
  plt.title(f"Random Sample of {num_random} Tokens")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  plt.savefig("figs/random_tokens.png", dpi=300)
  plt.close()


def main():
  corpus_path = Path("data/output.txt")
  token_dataset_path = Path("data/token_dataset.pt")
  vocab_to_ind_path = Path("data/vocab_to_ind.json")
  token_count_path = Path("data/token_count.json")
  corpus = load_corpus(corpus_path)
  tokenizer = CharTokenizer()
  tokens = tokenizer.tokenize_play(corpus)
  vocab_to_ind, token_counts = build_vocabulary(tokens)
  token_dataset = build_token_tensor(tokens, vocab_to_ind)
  plot_token_counts(token_counts)

  torch.save(token_dataset, token_dataset_path)
  with open(vocab_to_ind_path, "w", encoding="utf-8") as file:
    json.dump(vocab_to_ind, file, indent=4)
  with open(token_count_path, "w", encoding="utf-8") as file:
    json.dump(token_counts, file, indent=4)

  counts = torch.tensor(list(token_counts.values()), dtype=torch.int32)
  print("Summary statistics:")
  print(f"Token number: {len(token_counts)}")
  print(f"Corpuse character count: 2242619")
  print("Tokens appearing once:", (counts == 1).sum())
  print("Tokens appearing <= 5:", (counts <= 5).sum())
  print("Tokens appearing <= 10:", (counts <= 10).sum())
  print("Tokens appearing >= 100:", (counts >= 100).sum())
  print("Tokens appearing >= 1000:", (counts >= 1000).sum())


def sample_dataset():
  """Sample 100 tokens from token dataset to sanity-check the reverse process."""
  token_dataset_path = Path("data/token_dataset.pt")
  vocab_to_ind_path = Path("data/vocab_to_ind.json")

  dataset = torch.load(token_dataset_path, weights_only=True)
  with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
    vocab_to_ind = json.load(file)
  ind_to_vocab = {v: k for k, v in vocab_to_ind.items()}

  beginning = random.randint(0, len(dataset) - 101)

  txt = ""
  for i in range(beginning, beginning+100):
    txt += ind_to_vocab[int(dataset[i])]
  print(txt)


if __name__ == "__main__":
  corpus_path = Path("data/output.txt")
  main()
  sample_dataset()