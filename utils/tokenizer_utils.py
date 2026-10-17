import torch

from data_config import (
  BPE_TOKENIZER_DATA_PATH,
  CHAR_TOKENIZER_DATA_PATH,
)
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.bpe_tokenizer import BPETokenizer


def decide_tokenizer_data_path(tokenizer):
  if tokenizer == "char":
    return CHAR_TOKENIZER_DATA_PATH
  elif tokenizer == "bpe":
    return BPE_TOKENIZER_DATA_PATH


def create_tokenizer(tokenizer, **kwargs):
  if tokenizer == "char":
    return CharTokenizer()
  elif tokenizer == "bpe":
    return BPETokenizer()