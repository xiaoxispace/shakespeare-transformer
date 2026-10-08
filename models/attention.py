import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, dmodel, num_heads, mask=False):
        super(MultiHeadAttention, self).__init__()
        assert dmodel % num_heads == 0
        self.dmodel = dmodel 
        self.num_heads = num_heads
        self.dk = dmodel // num_heads
        self.block_size = block_size

        self.WQ = nn.Linear(dmodel, dmodel)
        self.WK = nn.Linear(dmodel, dmodel)
        self.WV = nn.Linear(dmodel, dmodel)
        self.linear = nn.Linear(dmodel, dmodel)

        if mask:
            m = torch.triu(torch.ones(self.block_size, self.block_size), diagonal=1).bool()
            self.register_buffer('mask', m)
        else:
            self.register_buffer('mask', None)

    def _split_heads(self, x):
      # x: (B, T, dmodel) => (B, H, T, dk)
      B, T, _ = x.size()
      return x.view(B, T, self.num_heads, self.dk).transpose(1, 2)

    def forward(self, Q, K, V):
      # Q: (B, T, dmodel)
      B, T, dmodel = Q.size()

      Q = self._split_heads(self.WQ(Q))  # (B, H, T, dk)
      K = self._split_heads(self.WK(K))
      V = self._split_heads(self.WV(V))

      scores = Q @ K.transpose(-2, -1) / (self.dk**0.5) #(B, H, T, T)
      if self.mask is not None:
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))
      attn = torch.softmax(scores, dim=-1)
      output = attn @ V  # (B, H, T, dk)

      output = output.transpose(1, 2).contiguous().view(B, T, dmodel)
      return self.linear(output)


class AttentionLayer(nn.Module):
    def __init__(self, block_size, dmodel, num_heads, mask=False, attention_dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.block_size = block_size
        self.dmodel = dmodel
        self.num_heads = num_heads
        self.mask = mask

        self.attention = MultiHeadAttention(self.block_size, self.dmodel, self.num_heads, self.mask)
        self.layer_norm = nn.LayerNorm(self.dmodel)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, Q, K, V):
        # Q, K, V: (B, T, dmodel)
        output = self.attention(Q, K, V)
        output = self.dropout(output)
        output = self.layer_norm(Q + output)

        return output


class FeedForwardLayer(nn.Module):
    def __init__(self, dmodel, dropout=0.1, d_ff=2048):
        super(FeedForwardLayer, self).__init__()
        self.dmodel = dmodel

        self.fully_connected = nn.Sequential(
            nn.Linear(self.dmodel, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, self.dmodel)
        )
        self.layer_norm = nn.LayerNorm(self.dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, dmodel)
        output = self.fully_connected(x)
        output = self.dropout(output)
        output = self.layer_norm(output + x)
        return output