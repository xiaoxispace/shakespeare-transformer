"""
This file contains the general configurations for dataset paths, tensor paths, etc. under
the "data/" directory.
"""
from pathlib import Path

SHAKESPEARES_PLAY_DIR = Path("data/shakespeares-works_TXT_FolgerShakespeare")
PLAT_TEXT_OUTPUT_PATH = Path("data/output.txt")

TOKENIZER_DATASET_FILE_NAME = "token_dataset.pt"
TOKENIZER_VOCAB_TO_IND_FILE_NAME = "vocab_to_ind.json"
TOKENIZER_TOKEN_COUNT_FILE_NAME = "token_count.json"

CHAR_TOKENIZER_DATA_PATH = Path("data/char_tokenizer")
BPE_TOKENIZER_DATA_PATH = Path("data/bpe_tokenizer")