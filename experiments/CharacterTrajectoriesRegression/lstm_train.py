import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / '../../'))

import argparse
import torch
import torch.nn as nn
import numpy as np
from src.models.LSTMEncoderDecoder import LSTMEncoderDecoder
from src.data.transforms import preprocess_for_transformer

# Set up matplotlib
import matplotlib.pyplot as plt

# Define hyperparameters
HP = {
    'log_dir': '/logs',
    'data_path': '../../data/processed/CharacterTrajectories/regression/30',
    'epochs': 500,
    'lr': 1e-3,
    'batch_size': 32,
    'input_channels': 4,
    'hidden_channels': 128,
    'output_channels': 3,
    'n_layers': 2
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
    with open(f'./{HP["log_dir"]}/log_lstm.txt', 'w') as f:
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

    # Get mask of y (where all values are zero)
    mask = y == 0
    mask = ~mask

    return y, mask

def getLenghts(inputs):
    # inputs is a tensor of size (batch, sequence_length, input_channels)
    # returns a list of size batch, indicating the true sequence length of each batch
    lengths = []
    for i in range(inputs.shape[0]):
        for j in range(1,inputs.shape[1]+1):
            if j == inputs.shape[1] or inputs[i][j][0] == 0:
                lengths.append(j)
                break

    return torch.tensor(lengths, dtype=torch.long).cpu()

def plot_trajectory(pred_y, batch_y):
    id_range = pred_y.size(0)
    id = np.random.randint(0, id_range)
    # Plot predicted trajectory and ground truth trajectory
    pred_vx, pred_vy, pred_f = [pred_y[id, : , i].cpu().numpy() for i in range(3)]
    true_vx, true_vy, true_f = [batch_y[id, : , i].cpu().numpy() for i in range(3)]

    # Integrate velocities to get positions
    pred_x, pred_y = np.cumsum(pred_vx), np.cumsum(pred_vy)
    true_x, true_y = np.cumsum(true_vx), np.cumsum(true_vy)

    # Remove last element and insert 0 to the beginning
    pred_x, pred_y = np.insert(pred_x[:-1], 0, 0), np.insert(pred_y[:-1], 0, 0)
    true_x, true_y = np.insert(true_x[:-1], 0, 0), np.insert(true_y[:-1], 0, 0)

    # Handle NaN values by only considering non-NaN values for normalization
    min_value, max_value = np.nanmin(pred_f), np.nanmax(pred_f)
    pred_f_normal = (pred_f - min_value) / (max_value - min_value)
    min_value, max_value = np.nanmin(true_f), np.nanmax(true_f)
    true_f_normal = (true_f - min_value) / (max_value - min_value)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.plot(pred_x, pred_y, label='Predicted trajectory', color='blue')
    plt.plot(true_x, true_y, label='True trajectory', color='red')
    plt.scatter(pred_x, pred_y, c=pred_f_normal, cmap='viridis', label='Predicted force')
    plt.scatter(true_x, true_y, c=true_f_normal, cmap='viridis', label='True force')
    plt.colorbar()
    plt.legend()
    plt.title('Predicted and true trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save the plot
    plt.savefig(f'./{HP["log_dir"]}/trajectory_lstm.png')

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
            batch_x, batch_y = batch
            batch_x, batch_y  = batch_x.to(device), batch_y.to(device)

            lengths = getLenghts(batch_x)

            # Get predictions
            pred_y = model(batch_x, lengths)
            pred_y = pred_y.squeeze(-1)

            # Get mask of batch_y (where all values are zero)
            batch_y, mask = strip_y(batch_y)

            # Apply mask to pred_y
            pred_y = pred_y * mask

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
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                lengths = getLenghts(batch_x)

                # Get predictions
                pred_y = model(batch_x, lengths)
                pred_y = pred_y.squeeze(-1)

                # Get mask of batch_y (where all values are zero)
                batch_y, mask = strip_y(batch_y)

                # Apply mask to pred_y
                pred_y = pred_y * mask

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
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            lengths = getLenghts(batch_x)
        
            # Get predictions
            pred_y = model(batch_x, lengths)
            pred_y = pred_y.squeeze(-1)

            # Get mask of batch_y (where all values are zero)
            batch_y, mask = strip_y(batch_y)

            # Apply mask to pred_y
            pred_y = pred_y * mask

            # Get loss
            test_loss = criterion(pred_y, batch_y) / len(batch_y)

            test_losses.append(test_loss.item())

        print('Test loss: {}'.format(np.mean(test_losses)))

    return np.mean(test_losses)
    
def main(device):
    # Load dataset
    X_train = torch.load(f'{HP["data_path"]}/X_train.pt')
    y_train = torch.load(f'{HP["data_path"]}/y_train.pt')
    X_test = torch.load(f'{HP["data_path"]}/X_test.pt')
    y_test = torch.load(f'{HP["data_path"]}/y_test.pt')

    # Preprocess data
    X_train, _ = preprocess_for_transformer(X_train)
    y_train, _ = preprocess_for_transformer(y_train)
    X_test, _ = preprocess_for_transformer(X_test)

    # Change to float32
    X_train = X_train.float()
    y_train = y_train.float()
    X_test = X_test.float()
    y_test = y_test.float()

    # Split train set into train and validation
    val_size = int(0.2 * len(X_train))
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    # Set up train dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=HP['batch_size'])

    # Set up validation dataloader
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=HP['batch_size'])

    # Set up test dataloader
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=HP['batch_size'])

    # Define model
    model = LSTMEncoderDecoder(HP['input_channels'], HP['hidden_channels'], HP['n_layers'], HP['output_channels'])
    model.to(device)

    # Define loss function
    criterion = nn.functional.mse_loss

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'])

    history, best_model = train_loop(model, criterion, optimizer, train_dataloader, val_dataloader, device)
    test_loss = evaluate(best_model, criterion, test_dataloader, device)
    logger(history, test_loss)

    # Save model
    torch.save(best_model.state_dict(), f'./{HP["log_dir"]}/lstm_model.pth')


if __name__ == '__main__':
    args = parse_args()
    device = set_gpu_device(args.gpu_id)
    main(device)