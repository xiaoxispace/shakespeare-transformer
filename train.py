
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from tqdm import tqdm
from pathlib import Path

from dataset import prepare_dataset_loader
from hyperparams import *
from models.transformer import Transformer


def train(model, train_loader, test_loader, criterion, optimizer, epochs=1, device='cuda'):
  # Define loss function and optimizer
  # Training loop
  train_loss_history = []
  test_loss_history = []
  val_loss_history = []
  epoch_sequence = []

  for epoch in range(epochs):
    # training
    log_registry = {}
    training_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
      model.train()
      for batch in train_loader:
        inputs, labels = batch[0].contiguous(), batch[1].contiguous()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients
        logits = model(inputs, None)  # Forward pass: (B, T, Emb)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        labels = labels.view(B * T)

        loss = criterion(logits, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        training_loss += loss.item() * B
        pbar.update(1)  # Update the progress bar
    train_loss_history.append(training_loss / len(train_loader.dataset))
    log_registry['train_loss'] = training_loss / len(train_loader.dataset)

    # testing
    with torch.no_grad():
      test_loss = 0.0
      model.eval()
      for batch in test_loader:
        inputs, labels = batch[0].contiguous(), batch[1].contiguous()
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs, None)  # Forward pass: (B, T, Emb)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        labels = labels.view(B * T)

        loss = criterion(logits, labels)  # Compute the loss
        test_loss += loss.item() * B
      
      val_loss = 0.0
      for batch in val_loader:
        inputs, labels = batch[0].contiguous(), batch[1].contiguous()
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs, None)  # Forward pass: (B, T, Emb)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        labels = labels.view(B * T)

        loss = criterion(logits, labels)  # Compute the loss
        val_loss += loss.item() * B

    test_loss_history.append(test_loss / len(test_loader.dataset))
    log_registry['test_loss'] = test_loss / len(test_loader.dataset)
    val_loss_history.append(val_loss / len(val_loader.dataset))
    log_registry['val_loss'] = val_loss / len(val_loader.dataset)
    wandb.log(log_registry)

    epoch_sequence.append(epoch + 1)

    print(f'Epoch {epoch + 1}/{epochs}: train loss {train_loss_history[-1]}, test loss {test_loss_history[-1]}')

    if epoch % 5 == 0 or epoch == epochs-1:
      # Save the model
      torch.save(model.state_dict(), f'./data/model-{epoch}.pth')

  print('Training complete!')

  return model 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
                  prog='shakespeare-training',
                  description='pretrain shakespeare transformer')
  parser.add_argument('-e', '--epochs', default=100, type=int)           # positional argument

  args = parser.parse_args()
  epochs = args.epochs
  enable_wandb = True

  train_end = 0.7
  val_end = 0.9

  if enable_wandb:
    wandb.init(
      project="shakespeare-transformer",
      config={
        "learning_rate": learning_rate, 
        "architecture": "Shakespear's transformer",
        "dataset": "Shakespeare",
        "epochs": epochs,
        "batch_size": batch_size,
        "block_size": block_size,
        "dmodel": dmodel,
        "dropout": dropout,
      }
    )
  else:
      print("Disable wandb")
      wandb.init(mode="disabled")

  print("Hello World!")
  print("CUDA available: ", torch.cuda.is_available())
  print("CUDA device count: ", torch.cuda.device_count())
  print("Epochs: ", epochs)

  token_dataset_path = Path("data/token_dataset.pt")
  token_sequence = torch.load(token_dataset_path)
  token_sequence = torch.tensor(token_sequence, dtype=torch.long)

  vocab_to_ind_path = Path("data/vocab_to_ind.json")
  with open(vocab_to_ind_path, "r", encoding="utf-8") as file:
    vocab_to_ind = json.load(file)
  ind_to_vocab = {v: k for k, v in vocab_to_ind.items()}
  vocab_size = len(vocab_to_ind)

  train_loader, val_loader, test_loader = prepare_dataset_loader(token_sequence, block_size, batch_size, train_end, val_end)
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

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  model = train(model, train_loader, test_loader, criterion, optimizer, epochs=epochs, device=device)


