import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / '../../'))

import torch
import torch.nn as nn
import torchcde
import numpy as np
from src.models.NeuralCDE import NeuralCDE
from src.data.cde_transforms import insert_random_missingness, fill_forward

# Define hyperparameters
HP = {
    'log_dir': '/logs/ncde_chartraj_classification',
    'data_path': '../../data/processed/CharacterTrajectories/classification',
    'missing_rate': 0.75,
    'epochs': 100,
    'lr': 1e-3,
    'batch_size': 32,
    'input_channels': 5,
    'hidden_channels': 32,
    'output_channels': 20,
    'hidden_layers': 3,
    'method': 'rk4',
    'step_size': 1
}

def logger(train_stats, test_acc):
    with open(f'./{HP["log_dir"]}/log_ncde_{int(HP["missing_rate"]*100)}.txt', 'w') as f:
        f.write('Hyperparameters\n')
        for key, value in HP.items():
            f.write(f'{key}: {value}\n')
        f.write('\n\n')
        f.write('History\n')
        for key, value in train_stats.items():
            f.write(f'{key}: {value}\n')
        f.write('\n\n')
        f.write(f'Test accuracy: {test_acc}\n')

def train_loop(model, criterion, optimizer, train_dataloader, val_dataloader, device):

    # Create history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_params = None

    for epoch in range(HP['epochs']):
        model.train()
        train_accs = []
        for i, batch in enumerate(train_dataloader):
            batch_coeffs, batch_y = batch
            batch_coeffs, batch_y = batch_coeffs.to(device), batch_y.to(device)

            # Get predictions
            pred_y = model(batch_coeffs)
            pred_y = pred_y.squeeze(-1)

            # Get loss
            loss = criterion(pred_y, batch_y.long())

            # Get accuracy
            pred_y = torch.argmax(pred_y, dim=1)
            train_acc = (pred_y == batch_y).sum().item() / len(batch_y)
            train_accs.append(train_acc)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()), 'Training accuracy: {}'.format(np.mean(train_accs)))

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
        
        # Update history
        history['train_loss'].append(loss.item())
        history['train_acc'].append(np.mean(train_accs))
        history['val_acc'].append(np.mean(val_accs))

        # Save best model
        if np.mean(val_accs) > best_val_acc:
            best_val_acc = np.mean(val_accs)
            best_params = model.state_dict()

    model.load_state_dict(best_params)

    return history, model

def evaluate(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        test_accs = []
        for i, batch in enumerate(test_dataloader):
            batch_coeffs, batch_y = batch
            batch_coeffs, batch_y = batch_coeffs.to(device), batch_y.to(device)

            # Get predictions
            pred_y = model(batch_coeffs)
            pred_y = pred_y.squeeze(-1)

            # Get accuracy
            pred_y = torch.argmax(pred_y, dim=1)

            test_acc = (pred_y == batch_y).sum().item() / len(batch_y)
            test_accs.append(test_acc)

        print('Test accuracy: {}'.format(np.mean(test_accs)))

    return np.mean(test_accs)
    
def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    X_train = torch.load(f'{HP["data_path"]}/X_train.pt')
    y_train = torch.load(f'{HP["data_path"]}/y_train.pt')
    X_test = torch.load(f'{HP["data_path"]}/X_test.pt')
    y_test = torch.load(f'{HP["data_path"]}/y_test.pt')

    # Insert random missingness
    X_train = insert_random_missingness(X_train, HP['missing_rate'])
    X_test = insert_random_missingness(X_test, HP['missing_rate'])

    # Fill forward missing values
    X_train = fill_forward(X_train)
    X_test = fill_forward(X_test)

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
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_train)
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=HP['batch_size'])

    # Set up validation dataloader
    val_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_val)
    val_dataset = torch.utils.data.TensorDataset(val_coeffs, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=HP['batch_size'])

    # Set up test dataloader
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_test)
    test_dataset = torch.utils.data.TensorDataset(test_coeffs, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=HP['batch_size'])

    # Define model
    model = NeuralCDE(HP['input_channels'], HP['hidden_channels'], HP['output_channels'])
    model.to(device)

    # Define loss function
    criterion = nn.functional.cross_entropy

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'])

    history, best_model = train_loop(model, criterion, optimizer, train_dataloader, val_dataloader, device)
    test_acc = evaluate(best_model, test_dataloader, device)
    logger(history, test_acc)


if __name__ == '__main__':
    main()