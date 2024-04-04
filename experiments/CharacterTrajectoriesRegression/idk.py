import torch

# Define hyperparameters
HP = {
    'log_dir': '/logs',
    'data_path': '../../data/processed/CharacterTrajectories/regression',
    'epochs': 100,
    'lr': 1e-3,
    'batch_size': 32,
    'input_channels': 4,
    'hidden_channels': 32,
    'output_channels': 3,
    'hidden_layers': 3,
    'method': 'rk4',
    'step_size': 1
}

X_train = torch.load(f'{HP["data_path"]}/X_train.pt')
y_train = torch.load(f'{HP["data_path"]}/y_train.pt')
X_test = torch.load(f'{HP["data_path"]}/X_test.pt')
y_test = torch.load(f'{HP["data_path"]}/y_test.pt')

print(X_train.shape)