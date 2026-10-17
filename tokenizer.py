"""
This file takes output.txt, the concatenated play corpus, and then tokenize it. 
The tokenizer groups consecutive characters of the same type (alpha, digit, or else).
Executing this file produces 
  1. tokenizer mapping in json, which is a 1-to-1 mapping from string to integer and
  2. token index of the corpus stored in pytorch tensor of type torch.int32
  3. a json string-to-integer map that denotes each token's number of occurrences. 
"""
import argparse
import logging
import random
import matplotlib.pyplot as plt

from data_config import PLAT_TEXT_OUTPUT_PATH
from utils.tokenizer_utils import create_tokenizer, decide_tokenizer_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tokenizer")


def load_corpus(corpus_path):
  with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = f.read()
  return corpus


def display_token(token):
  """Normalize displayed tokens on frequency figures"""
  if token == " ":
    return "<space>"
  elif token == "\n":
    return "<newline>"
  elif token == "\t":
    return "<tab>"
  elif token == "\r":
    return "<return>"
  else:
    return token


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
  displayed_tokens = [display_token(token) for token, _ in tokens]

  plt.figure(figsize=(12, 5))
  plt.bar(displayed_tokens, counts)
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
    [display_token(token) for token in token_counts.keys()],
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
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-t', '--tokenizer', choices=["char", "bpe"], default='char')

  parser = argparser.parse_args()
  tokenizer = parser.tokenizer

  tokenizer_data_path = decide_tokenizer_data_path(tokenizer)

  corpus_path = PLAT_TEXT_OUTPUT_PATH
  corpus = load_corpus(corpus_path)

  tokenizer = create_tokenizer(tokenizer)
  tokenizer.train(corpus)
  tokenizer.save(tokenizer_data_path)
  plot_token_counts(tokenizer.token_counts)


# def sample_dataset():
#   """Sample 100 tokens from token dataset to sanity-check the reverse process."""
#   token_path = Path("data/token_dataset.pt")
#   vocab_to_ind_path = Path("data/vocab_to_ind.json")

#   dataset = torch.load(token_path, weights_only=True)
#   with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
#     vocab_to_ind = json.load(file)
#   ind_to_vocab = {v: k for k, v in vocab_to_ind.items()}

#   beginning = random.randint(0, len(dataset) - 101)

#   txt = ""
#   for i in range(beginning, beginning+100):
#     txt += ind_to_vocab[int(dataset[i])]
#   print(txt)


if __name__ == "__main__":
  main()