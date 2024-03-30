import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / '../'))

import math
import torch
import torch.nn as nn
import torchcde
import numpy as np
from src.models.NeuralCDE import NeuralCDE
from src.data.cde_transforms import insert_random_missingness, fill_forward

# Define hyperparameters
HP = {
    'data_path': 'data/processed/CharacterTrajectories/classification',
    'epochs': 100,
    'lr': 1e-4,
    'batch_size': 64,
    'input_channels': 5,
    'hidden_channels': 64,
    'output_channels': 1,
}

def main():
    # Load dataset
    X_train = torch.load(f'{HP["data_path"]}/X_train.pt')
    y_train = torch.load(f'{HP["data_path"]}/y_train.pt')
    X_val = torch.load(f'{HP["data_path"]}/X_test.pt')
    y_val = torch.load(f'{HP["data_path"]}/y_test.pt')

    # Insert random missingness
    X_train = insert_random_missingness(X_train)
    X_val = insert_random_missingness(X_val)

    # Fill forward missing values
    X_train = fill_forward(X_train)
    X_val = fill_forward(X_val)

    # Change to float32
    X_train = X_train.float()

    # Define model
    model = NeuralCDE(HP['input_channels'], HP['hidden_channels'], HP['output_channels'])

    # Define loss function
    criterion = nn.functional.binary_cross_entropy_with_logits

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'])

    # Train model
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train)
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=HP['batch_size'])

    for epoch in range(HP['epochs']):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))


if __name__ == '__main__':
    main()