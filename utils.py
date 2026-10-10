"""This file contains the utilities for output evaluations and generations."""
import torch


def generate(model, token_indx, length=1000, device='cpu'):
  """Generate the list of output token indices given a list or a batch of list of token indices. """
  generated_token_index = []

  with torch.no_grad():
    for _ in range(length):
      inputs = torch.tensor(token_indx).to(device)
      if len(inputs.size()) == 1:  # input is only a list
        inputs = inputs.unsqueeze(0)

      output = model(inputs, inputs) # (1, T, VOCAB)
      output = output[:, -1, :] # (1, VOCAB)
      output = torch.softmax(output, dim=-1) # [1, VOCAB]
      output = torch.multinomial(output, num_samples=1)

      item = output.item()
      generated_token_index.append(item)
      token_indx.append(item)

  return generated_token_index