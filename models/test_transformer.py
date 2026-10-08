import torch

from transformer import PositionalEncoding, Encoder, Decoder, Transformer


def test_pos_encoding():
  batch = 6
  block_size = 15
  dmodel = 12
  pe = PositionalEncoding(block_size, dmodel)
  x = torch.zeros((batch, block_size, dmodel))
  y = pe(x)
  assert x.shape == y.shape

def test_encoder():
  batch = 6
  block_size = 15
  dmodel = 12
  dropout=0.5
  num_of_heads=3
  encoder = Encoder(block_size, dropout, dmodel, num_of_heads)

  x = torch.zeros((batch, block_size, dmodel))
  y = encoder(x)
  assert x.shape == y.shape

def test_decoder():
  batch = 6
  block_size = 15
  dmodel = 12
  dropout=0.5
  num_of_heads=3

  x = torch.zeros((batch, block_size, dmodel))
  decoder = Decoder(block_size, dropout, dmodel, num_of_heads)
  y = decoder(x, x)
  assert x.shape == y.shape

  decoder = Decoder(block_size, dropout, dmodel, num_of_heads, decoder_only=True)
  y = decoder(x)
  assert x.shape == y.shape


def test_encoder_decoder_transformer():
  vocab_size = 100
  batch = 6
  seq_len = 15
  block_size = 36
  dmodel = 12
  dropout = 0.5
  num_of_heads = 3
  num_of_decoder_layers = 3
  num_of_encoder_layers = 3

  t = Transformer(vocab_size, block_size, dropout, dmodel, num_of_encoder_layers, num_of_decoder_layers, num_of_heads)
  x = torch.randint(0, vocab_size, (batch, seq_len))
  y = t(x, x)

  assert x.shape + (vocab_size,) == y.shape


def test_decoder_only_transformer():
  vocab_size = 100
  batch = 6
  seq_len = 15
  block_size = 36
  dmodel = 12
  dropout = 0.5
  num_of_heads = 3
  num_of_decoder_layers = 3
  num_of_encoder_layers = 3

  t = Transformer(vocab_size, block_size, dropout, dmodel, num_of_encoder_layers, num_of_decoder_layers, num_of_heads, decoder_only=True)
  x = torch.randint(0, vocab_size, (batch, seq_len))
  y = t(x, None)

  assert x.shape + (vocab_size,) == y.shape