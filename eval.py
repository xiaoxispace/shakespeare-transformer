"""This file reports basic NLP metrics including perplexity."""
import argparse
import json
import torch

from dataset import prepare_dataset_loader
from hyperparams import *
from models.transformer import Transformer
from pathlib import Path
from utils import perplexity


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-m', '--model-path', type=str)

  parser = argparser.parse_args()
  model_path = Path(parser.model_path)

  train_end = 0.7  # same as train.py
  val_end = 0.9  # same as train.py

  block_size = config['block_size']
  batch_size = config['batch_size']
  decoder_only = config['decoder_only']
  num_of_encoder_layers = config['num_of_encoder_layers']
  num_of_decoder_layers = config['num_of_decoder_layers']
  num_of_heads = config['num_of_heads']
  dmodel = config['dmodel']
  dropout = config['dropout']
  device = config['device']

  print("Loading vocab ...")
  token_dataset_path = Path("data/token_dataset.pt")
  vocab_to_ind_path = Path("data/vocab_to_ind.json")
  token_count_path = Path("data/token_count.json")
  with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
    token_sequence = torch.load(token_dataset_path).to(torch.long)
  with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
    vocab_to_ind = json.load(file)
  ind_to_vocab = {v: k for k, v in vocab_to_ind.items()}
  vocab_size = len(vocab_to_ind)
  print("Vocab size: ", len(vocab_to_ind))
  print("Done loading vocab ...")

  print("Construct and load model ...")
  model = Transformer(
    vocab_size,
    block_size,
    dropout,
    dmodel,
    num_of_encoder_layers,
    num_of_decoder_layers,
    num_of_heads,
    decoder_only=decoder_only
  ).to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  print("Construct and load model done.")

  print("Load dataset to compute metrics.")
  token_dataset_path = Path("data/token_dataset.pt")
  token_sequence = torch.load(token_dataset_path).to(torch.long)
  _, _, test_loader = prepare_dataset_loader(token_sequence, block_size, batch_size, train_end, val_end)
  print("Load done.\n")

  per = perplexity(model, test_loader, device)
  print(f"Perlexity: {per}")