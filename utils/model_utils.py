import torch
import json
import os

from pathlib import Path

from models.transformer import Transformer
from utils.tokenizer_utils import load_tokenizer


def make_model_saver(config, vocab_to_ind, run_name=None, save_dir="checkpoints"):
    def save_model(model, epoch, is_best_model=False):
      if model.decoder_only:
        filename = (
          f"_ctx{config['block_size']}"
          f"_d{config['dmodel']}"
          f"_L{config['num_of_decoder_layers']}"
          f"_H{config['num_of_heads']}"
          f"_lr{config['learning_rate']:.0e}"
          f"_dropout{config['dropout']:.0e}"
          f"_e{epoch}.pth"
        )
      else:
        filename = (
          f"_ctx{config['block_size']}"
          f"_d{config['dmodel']}"
          f"_dcdr{config['num_of_decoder_layers']}"
          f"_ecdr{config['num_of_encoder_layers']}"
          f"_H{config['num_of_heads']}"
          f"_lr{config['learning_rate']:.0e}"
          f"_dropout{config['dropout']:.0e}"
          f"_e{epoch}.pth"
        )

      if is_best_model:
        filename = "best_model" + filename
      else:
        filename = "model" + filename

      if run_name:
        run_dir = os.path.join(save_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(run_dir, filename)
        vocab_to_ind_path = os.path.join(run_dir, 'vocab_to_ind.json')
        model_config_path = os.path.join(run_dir, 'config.json')
      else:
        model_path = os.path.join(save_dir, filename)
        vocab_to_ind_path = os.path.join(save_dir, 'vocab_to_ind.json')
        model_config_path = os.path.join(save_dir, 'config.json')

      if not os.path.isfile(model_config_path):
        with open(model_config_path, "w", encoding="utf-8") as f:
          json.dump(config, f, indent=2)
        print(f"Model config saved to {model_config_path}")

      if not os.path.isfile(vocab_to_ind_path):
        with open(vocab_to_ind_path, "w", encoding="utf-8") as f:
          json.dump(vocab_to_ind, f, indent=2)
        print(f"Model vocab_to_ind saved to {vocab_to_ind_path}")

      torch.save(model.state_dict(), model_path)
      print(f"Model saved to {model_path}")

    return save_model


def load_model(model_path, config='config.json', tokenizer_path="tokenizer"):
  model_dir = Path(model_path).parent
  tokenizer_path = model_dir / tokenizer_path
  config_path = model_dir / config

  with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

  tokenizer_name = config["tokenizer"]
  tokenizer = load_tokenizer(tokenizer_name, tokenizer_path)

  model = Transformer(**config)
  model.load_state_dict(torch.load(model_path))

  return model, tokenizer, config