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
    'data_path': '../data/processed/CharacterTrajectories/classification',
    'epochs': 1000,
    'lr': 1e-3,
    'batch_size': 32,
    'input_channels': 5,
    'hidden_channels': 32,
    'output_channels': 20,
}

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    y_train = y_train.float()
    X_val = X_val.float()
    y_val = y_val.float()

    # Define model
    model = NeuralCDE(HP['input_channels'], HP['hidden_channels'], HP['output_channels'])
    model.to(device)

    # Define loss function
    criterion = nn.functional.cross_entropy

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'])

    # Set up train dataloader
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train)
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=HP['batch_size'])

    # Set up validation dataloader
    val_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_val)
    val_dataset = torch.utils.data.TensorDataset(val_coeffs, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=HP['batch_size'])

    for epoch in range(HP['epochs']):
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch_coeffs, batch_y = batch
            batch_coeffs, batch_y = batch_coeffs.to(device), batch_y.to(device)

            # Get predictions
            pred_y = model(batch_coeffs)
            pred_y = pred_y.squeeze(-1)

            # Get loss
            loss = criterion(pred_y, batch_y.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

        if epoch % 10 == 0:
            model.eval()

            with torch.no_grad():
                val_accs = []
                for i, batch in enumerate(val_dataloader):
                    batch_coeffs, batch_y = batch
                    batch_coeffs, batch_y = batch_coeffs.to(device), batch_y.to(device)

                    # Get predictions
                    pred_y = model(batch_coeffs)
                    pred_y = pred_y.squeeze(-1)

                    # Get accuracy
                    pred_y = torch.argmax(pred_y, dim=1)

                    val_acc = (pred_y == batch_y).sum().item() / len(batch_y)
                    val_accs.append(val_acc)

                print('Validation accuracy: {}'.format(np.mean(val_accs)))
            
            model.train()


if __name__ == '__main__':
    main()