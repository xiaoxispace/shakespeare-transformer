import torch


config = {
  "block_size": 256,
  "batch_size": 256,
  "dmodel": 256,
  "learning_rate": 3e-4,
  "cuda_available": torch.cuda.is_available(),
  "dropout": 0.1,
  "device": "cuda" if torch.cuda.is_available() else "cpu",
  "num_of_decoder_layers": 3,
  "num_of_encoder_layers": 3,
  "num_of_heads": 4,
  "decoder_only": True
}