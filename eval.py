import argparse

import json
import torch

from hyperparams import *
from models.transformer import Transformer
from pathlib import Path


if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-m', '--model-path', type=str)
  argparser.add_argument('-l', '--length', type=int, default=1000)
  argparser.add_argument('-s', '--start', type=str, default="to be")

  parser = argparser.parse_args()
  model_path = Path(parser.model_path)
  length = parser.length
  start = parser.start

  block_size = config['block_size']
  batch_size = config['batch_size']
  decoder_only = config['decoder_only']
  num_of_encoder_layers = config['num_of_encoder_layers']
  num_of_decoder_layers = config['num_of_decoder_layers']
  num_of_heads = config['num_of_heads']
  dmodel = config['dmodel']
  dropout = config['dropout']
  learning_rate = config['learning_rate']
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
  print()

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
  print("Construct and load model done.\n")

  token_indx = [
    vocab_to_ind[c] for c in start
  ]

  print("================================")
  print("|| Begin generating contents: ||")
  print("================================")
  print()

  with torch.no_grad():
    for i in range(length):
      input = torch.tensor(token_indx).unsqueeze(0).to(device)
      output = model(input, input)
      output = output[:, -1, :]
      output = torch.softmax(output, dim=-1) #[1, vocab_size]
      output = torch.multinomial(output, num_samples=1)
      token_indx.append(output.item())
      print(ind_to_vocab[output.item()], end='')
  