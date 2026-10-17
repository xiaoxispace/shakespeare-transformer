import json
import torch

from collections import Counter

from data_config import (
  TOKENIZER_RAW_TOKEN_FILE_NAME,
  TOKENIZER_TOKEN_COUNT_FILE_NAME,
  TOKENIZER_VOCAB_TO_IND_FILE_NAME
)
from tokenizers.tokenizer_abc import TokenizerABC


def prepare_tokenizer_data_paths(tokenizer_data_path):
  tokenizer_data_path.mkdir(parents=True, exist_ok=True)
  raw_token_path = tokenizer_data_path / TOKENIZER_RAW_TOKEN_FILE_NAME
  vocab_to_ind_path = tokenizer_data_path / TOKENIZER_VOCAB_TO_IND_FILE_NAME
  token_count_path = tokenizer_data_path / TOKENIZER_TOKEN_COUNT_FILE_NAME
  return raw_token_path, vocab_to_ind_path, token_count_path


class CharTokenizer(TokenizerABC):
  def __init__(self):
    super(CharTokenizer).__init__()
    self.tokens = None
    self.vocab_to_ind = None
    self.token_counts = None

  def train(self, corpus, **kwargs):
    self.tokens = self._tokenize_play(corpus)
    self.vocab_to_ind, self.token_counts = self._build_vocabulary(self.tokens)
    counts = torch.tensor(list(self.token_counts.values()), dtype=torch.int32)
    print("Summary statistics:")
    print(f"Token number: {len(self.token_counts)}")
    print(f"Corpuse character count: {len(self.tokens)}")
    print("Tokens appearing once:", (counts == 1).sum())
    print("Tokens appearing <= 5:", (counts <= 5).sum())
    print("Tokens appearing <= 10:", (counts <= 10).sum())
    print("Tokens appearing >= 100:", (counts >= 100).sum())
    print("Tokens appearing >= 1000:", (counts >= 1000).sum())

  def save(self, path):
    raw_token_path, vocab_to_ind_path, token_count_path = prepare_tokenizer_data_paths(path)
    with open(raw_token_path, "w", encoding="utf-8") as file:
      json.dump(self.tokens, file)
    with open(vocab_to_ind_path, "w", encoding="utf-8") as file:
      json.dump(self.vocab_to_ind, file, indent=4)
    with open(token_count_path, "w", encoding="utf-8") as file:
      json.dump(self.token_counts, file, indent=4)

  @classmethod
  def load(cls, path):
    raw_token_path, vocab_to_ind_path, token_count_path = prepare_tokenizer_data_paths(path)
    with open(raw_token_path, "r", encoding="utf-8") as file:
      tokens = json.load(file)
    with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
      vocab_to_ind = json.load(file)
    with open(token_count_path, "r", encoding="utf-8") as file:
      token_counts = json.load(file)
    t = CharTokenizer()
    t.tokens = tokens
    t.vocab_to_ind = vocab_to_ind
    t.token_counts = token_counts
    return t

  def encode(self, text):
    pass

  def decode(self, tokens):
    pass

  def _tokenize_play(self, play_string):
    """Tokenize text at character level."""
    tokens = []
    i = 0
    while i < len(play_string):
      c = play_string[i]
      tokens.append(c)
      i += 1
    return tokens

  def _build_vocabulary(self, tokens):
    """Build token-to-index and token-count dictionaries."""
    token_counts = Counter(tokens)

    # Sort for deterministic vocabulary construction.
    vocab = sorted(token_counts, reverse=True)

    vocab_to_ind = {
      token: i
      for i, token in enumerate(vocab)
    }

    return vocab_to_ind, dict(token_counts)




