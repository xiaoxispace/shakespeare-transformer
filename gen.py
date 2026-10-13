import argparse
import json
import torch

from pathlib import Path

from utils.eval_utils import generate
from utils.model_utils import load_model


if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-m', '--model-path', type=str)
  argparser.add_argument('-l', '--length', type=int, default=1000)
  argparser.add_argument('-s', '--start', type=str, default="to be")

  parser = argparser.parse_args()
  model_path = Path(parser.model_path)
  length = parser.length
  start = parser.start
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model, vocab_to_ind, config = load_model(model_path)
  model.to(device)
  model.eval()
  config_display = {k: str(v) for k, v in config.items()}
  print("Model config: ", json.dumps(config_display, indent=2))

  ind_to_vocab = {v: k for k, v in vocab_to_ind.items()}
  token_indx = [vocab_to_ind[c] for c in start]

  print("|| Begin generating contents: ||")
  tokens = generate(model, token_indx, device=device)
  outputs = "".join([ind_to_vocab[t] for t in tokens])
  print(outputs)
