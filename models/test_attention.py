import torch

from attention import MultiHeadAttention


def test_mha_forward():
  (B, T, dmodel) = (3, 6, 12)
  block_size = 15
  num_heads = 3
  x = torch.rand(B, T, dmodel)
  mha = MultiHeadAttention(
    block_size, dmodel, num_heads
  )
  out = mha(x, x, x)
  assert out.shape == (B, T, dmodel)


def test_mha_mask():
  (B, T, dmodel) = (3, 6, 12)
  block_size = 15
  num_heads = 3

  mha = MultiHeadAttention(
    block_size, dmodel, num_heads
  )
  masked_mha = MultiHeadAttention(
    block_size, dmodel, num_heads, mask=True
  )

  mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()

  assert mha.mask is None
  assert masked_mha.mask is not None
  assert torch.equal(masked_mha.mask, mask)


