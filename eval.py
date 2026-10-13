"""This file reports basic NLP metrics including perplexity."""
import argparse
import torch

from dataset import prepare_dataset_loader
from pathlib import Path
from utils.eval_utils import perplexity
from utils.model_utils import load_model


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-m', '--model-path', type=str)

  parser = argparser.parse_args()
  model_path = Path(parser.model_path)

  train_end = 0.7  # same as train.py
  val_end = 0.9  # same as train.py
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model, vocab_to_ind, config = load_model(model_path)
  model.to(device)

  print("Load dataset to compute metrics.")
  token_dataset_path = Path("data/token_dataset.pt")
  token_sequence = torch.load(token_dataset_path).to(torch.long)
  _, _, test_loader = prepare_dataset_loader(token_sequence, config["block_size"], config["batch_size"], train_end, val_end)
  print("Load done.\n")

  per = perplexity(model, test_loader, device)
  print(f"Perlexity: {per}")