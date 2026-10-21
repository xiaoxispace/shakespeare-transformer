"""
This file contains the general configurations for dataset paths, tensor paths, etc. under
the "data/" directory.
"""
from pathlib import Path

SHAKESPEARES_PLAY_DIR = Path("data/shakespeares-works_TXT_FolgerShakespeare")
PLAT_TEXT_OUTPUT_PATH = Path("data/output.txt")

DATASETS_DIR = Path("data/datasets")

TOKENIZER_VOCAB_TO_IND_FILE_NAME = "vocab_to_ind.json"
TOKENIZER_CONFIG_FILE_NAME = "config.json"

TOKENIZER_DATA_PATH = Path("data/tokenizers")