import torch

from data_config import BPE_TOKENIZER_DATA_PATH, CHAR_TOKENIZER_DATA_PATH
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.bpe_tokenizer import BPETokenizer


def decide_tokenizer_data_path(tokenizer):
  if tokenizer == "char":
    return CHAR_TOKENIZER_DATA_PATH
  elif tokenizer == "bpe":
    return BPE_TOKENIZER_DATA_PATH


def create_tokenizer(tokenizer_name, **kwargs):
  if tokenizer_name == "char":
    return CharTokenizer()
  elif tokenizer_name == "bpe":
    return BPETokenizer(threshold=kwargs["bpe_vocab_threshold"])


def load_tokenizer(tokenizer_name, tokenizer_path):
  if tokenizer_name == "char":
    return CharTokenizer.load(tokenizer_path)
  elif tokenizer_name == "bpe":
    return BPETokenizer.load(tokenizer_path)