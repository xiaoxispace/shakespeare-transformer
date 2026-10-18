import json
import torch

from collections import Counter
from pathlib import Path

from data_config import (
  TOKENIZER_RAW_TOKEN_FILE_NAME,
  TOKENIZER_TOKEN_COUNT_FILE_NAME,
  TOKENIZER_VOCAB_TO_IND_FILE_NAME
)
from tokenizers.tokenizer_abc import TokenizerABC


def prepare_tokenizer_data_paths(tokenizer_data_path):
  tokenizer_data_path = Path(tokenizer_data_path)
  tokenizer_data_path.mkdir(parents=True, exist_ok=True)
  raw_token_path = tokenizer_data_path / TOKENIZER_RAW_TOKEN_FILE_NAME
  vocab_to_ind_path = tokenizer_data_path / TOKENIZER_VOCAB_TO_IND_FILE_NAME
  token_count_path = tokenizer_data_path / TOKENIZER_TOKEN_COUNT_FILE_NAME
  return raw_token_path, vocab_to_ind_path, token_count_path


class CharTokenizer(TokenizerABC):
  def __init__(self):
    super(CharTokenizer).__init__()
    self.vocab_to_ind = None
    self.ind_to_vocab = None

  def train(self, corpus, **kwargs):
    raw_tokens = self._tokenize_play(corpus)
    self.vocab_to_ind, self.token_counts = self._build_vocabulary(raw_tokens)
    counts = torch.tensor(list(self.token_counts.values()), dtype=torch.int32)
    print("Summary statistics:")
    print(f"Token number: {len(self.token_counts)}")
    print(f"Corpuse raw token count: {len(raw_tokens)}")
    print("Tokens appearing once:", (counts == 1).sum())
    print("Tokens appearing <= 5:", (counts <= 5).sum())
    print("Tokens appearing <= 10:", (counts <= 10).sum())
    print("Tokens appearing >= 100:", (counts >= 100).sum())
    print("Tokens appearing >= 1000:", (counts >= 1000).sum())

  def save(self, path):
    _, vocab_to_ind_path, _ = prepare_tokenizer_data_paths(path)
    with open(vocab_to_ind_path, "w", encoding="utf-8") as file:
      json.dump(self.vocab_to_ind, file, indent=4)

  @classmethod
  def load(cls, path):
    _, vocab_to_ind_path, _ = prepare_tokenizer_data_paths(path)
    with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
      vocab_to_ind = json.load(file)
    t = cls()
    t.vocab_to_ind = vocab_to_ind
    return t

  def encode(self, text):
    return [self.vocab_to_ind[s] for s in text]

  def decode(self, token_ids):
    if not self.ind_to_vocab:
      self.ind_to_vocab = {v :k for k, v in self.vocab_to_ind.items()}
    return ''.join([self.ind_to_vocab[i] for i in token_ids])

  def vocab_size(self):
    return len(self.vocab_to_ind)

  def _tokenize_play(self, play_string):
    """Tokenize text at character level."""
    raw_tokens = []
    i = 0
    while i < len(play_string):
      c = play_string[i]
      raw_tokens.append(c)
      i += 1
    return raw_tokens

  def _build_vocabulary(self, raw_tokens):
    """Build token-to-index and token-count dictionaries."""
    token_counts = Counter(raw_tokens)

    # Sort for deterministic vocabulary construction.
    vocab = sorted(token_counts, reverse=True)

    vocab_to_ind = {
      token: i
      for i, token in enumerate(vocab)
    }

    return vocab_to_ind, dict(token_counts)




