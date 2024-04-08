import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / '../../'))

import math
import argparse
import torch
import torch.nn as nn
import numpy as np
from src.models.TransformerSeq import TransformerDecoder

# Set up matplotlib
import matplotlib.pyplot as plt

# Define hyperparameters
HP = {
    'log_dir': '/logs',
    'data_path': '../../data/processed/ERA5',
    'epochs': 250,
    'lr': 1e-3,
    'batch_size': 32,
    'input_channels': 2,
    'hidden_channels': 32,
    'output_channels': 2,
    'hidden_layers': 3,
    'n_heads': 4,
    'n_layers': 3,
    'dropout': 0.1
}

def parse_args():
    parser = argparse.ArgumentParser(description="Script to set GPU device.")
    parser.add_argument("--gpu_id", type=int, default=0, help="Index of the GPU device to use.")
    return parser.parse_args()

def set_gpu_device(gpu_id):
    """
    Set the GPU device to use.

    Parameters
    ----------
    gpu_id : int
        Index of the GPU device to use.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda', gpu_id)
    else:
        device = torch.device('cpu')
    return device

def logger(train_stats, test_loss):
    with open(f'./{HP["log_dir"]}/log_transformer.txt', 'w') as f:
        f.write('Hyperparameters\n')
        for key, value in HP.items():
            f.write(f'{key}: {value}\n')
        f.write('\n\n')
        f.write('History\n')
        for key, value in train_stats.items():
            f.write(f'{key}: {value}\n')
        f.write('\n\n')
        f.write(f'Test accuracy: {test_loss}\n')

def strip_y(y):
    # Strip time channel
    y = y[:, :, 1:]

    return y

def plot_trajectory(pred_y, batch_y):
    pred_y = pred_y.squeeze().detach().cpu().numpy()
    batch_y = batch_y.squeeze().detach().cpu().numpy()

    pred_temp = pred_y[:, 0]
    true_temp = batch_y[:, 0]
    pred_humidity = pred_y[:, 1]
    true_humidity = batch_y[:, 1]

    t = np.arange(len(pred_temp))

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Plot temperature
    axes[0].scatter(t, pred_temp, label='Predicted temperature', color='red')
    axes[0].scatter(t, true_temp, label='True temperature', color='blue')
    axes[0].plot(t, pred_temp, color='red', alpha=0.5)
    axes[0].plot(t, true_temp, color='blue', alpha=0.5)
    axes[0].set_xlabel('Bi-Monthly Time Step')
    axes[0].set_ylabel('Norm Value')
    axes[0].set_title('Temperature Trajectory')
    axes[0].legend()

    # Plot humidity
    axes[1].scatter(t, pred_humidity, label='Predicted humidity', color='green')
    axes[1].scatter(t, true_humidity, label='True humidity', color='orange')
    axes[1].plot(t, pred_humidity, color='green', alpha=0.5)
    axes[1].plot(t, true_humidity, color='orange', alpha=0.5)
    axes[1].set_xlabel('Bi-Monthly Time Step')
    axes[1].set_ylabel('Norm Value')
    axes[1].set_title('Humidity Trajectory')
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'./{HP["log_dir"]}/trajectory_transformer.png')

    # Explicitly close the figure
    plt.close()

def train_loop(model, criterion, optimizer, train_dataloader, val_dataloader, device):

    # Create history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = np.inf
    best_params = None

    for epoch in range(HP['epochs']):
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch_x = batch[0]
            batch_x = batch_x.to(device)

            # Strip time channel
            batch_y = strip_y(batch_x)
            y0 = batch_y[:, 0, :].unsqueeze(1)

            pred_y = model.autoregressive_predict(y0, max_len=23)

            # Get loss
            loss = criterion(pred_y, batch_y) / len(batch_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

        model.eval()

        with torch.no_grad():
            val_losses = []
            for i, batch in enumerate(val_dataloader):
                batch_x = batch[0]
                batch_x = batch_x.to(device)

                # Strip time channel
                batch_y = strip_y(batch_x)
                y0 = batch_y[:, 0, :].unsqueeze(1)

                pred_y = model.autoregressive_predict(y0, max_len=23)

                # Get loss
                val_loss = criterion(pred_y, batch_y) / len(batch_y)

                val_losses.append(val_loss.item())

            print('Validation loss: {}'.format(np.mean(val_losses)))

            if epoch % 10 == 0:
                plot_trajectory(pred_y, batch_y)
        
        # Update history
        history['train_loss'].append(loss.item())
        history['val_loss'].append(np.mean(val_losses))

        # Save best model
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            best_params = model.state_dict()

    model.load_state_dict(best_params)

    return history, model

def evaluate(model, criterion, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        test_losses = []
        for i, batch in enumerate(test_dataloader):
            batch_x = batch[0]
            batch_x = batch_x.to(device)

            # Strip time channel
            batch_y = strip_y(batch_x)
            y0 = batch_y[:, 0, :].unsqueeze(1)

            pred_y = model.autoregressive_predict(y0, max_len=23)        

            # Get loss
            test_loss = criterion(pred_y, batch_y.long()) / len(batch_y)

            test_losses.append(test_loss.item())

        print('Test loss: {}'.format(np.mean(test_losses)))

    return np.mean(test_losses)
    
def main(device):
    # Load dataset
    X_train = torch.load(f'{HP["data_path"]}/X_train.pt')
    X_test = torch.load(f'{HP["data_path"]}/X_test.pt')

    # Change to float32
    X_train = X_train.float()
    X_test = X_test.float()

    # Split train set into train and validation
    val_size = int(0.2 * len(X_train))
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]

    # Set up train dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=HP['batch_size'])

    # Set up validation dataloader
    val_dataset = torch.utils.data.TensorDataset(X_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=HP['batch_size'])

    # Set up test dataloader
    test_dataset = torch.utils.data.TensorDataset(X_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=HP['batch_size'])

    # Define model
    model = TransformerDecoder(HP['input_channels'], HP['hidden_channels'], HP['n_heads'], HP['n_layers'], HP['dropout'], HP['output_channels'])
    model.to(device)

    # Define loss function
    criterion = nn.functional.mse_loss

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'])

    history, best_model = train_loop(model, criterion, optimizer, train_dataloader, val_dataloader, device)
    test_loss = evaluate(best_model, criterion, test_dataloader, device)
    logger(history, test_loss)

    # Save model
    torch.save(best_model.state_dict(), f'./{HP["log_dir"]}/transformer_model.pth')


if __name__ == '__main__':
    args = parse_args()
    device = set_gpu_device(args.gpu_id)
    main(device)