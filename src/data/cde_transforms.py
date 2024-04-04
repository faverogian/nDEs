import math
import torch
import numpy as np

def insert_random_missingness(X, missing_rate=0.25):
    '''
    Insert random missingness into the data.

    Parameters
    ----------
    X : torch.Tensor
        The input data. Shape (n_samples, sequence_length, n_features)
    missing_rate : float
        The rate of missingness to insert in sequences of the samples. 
        Each sample in X will have this proportion of its sequence (each feature)
        set to NaN. Take sequence ending at point before zero padding.

    Returns
    -------
    out : torch.Tensor
        The input data with missingness inserted as NaNs across the datapoint.
        A cumulative mask channel is concatenated to the input data to indicate
        missingness. Shape (n_samples, sequence_length, n_features + 1)
    '''
    n_samples, sequence_length, _ = X.shape

    # Create a mask that zeros out where time is zero
    time_mask = X[:, :, 0] == 0
    time_mask[:, 0] = 0

    # Create a mask for missingness
    mask = torch.zeros(n_samples, sequence_length)
    for i in range(n_samples):
        unpadded_length = sequence_length - sum(time_mask[i])
        mask[i][:unpadded_length] = torch.randperm(unpadded_length) < int(missing_rate * unpadded_length)

    # Convert mask to boolean
    mask = mask.bool()
    mask = mask.unsqueeze(-1)

    # Create a cumulative mask channel 
    invert_mask = ~mask
    cum_mask = invert_mask.cumsum(dim=1).float()
    
    # Repeat the mask across all channels
    mask = mask.repeat(1, 1, X.shape[-1])

    # Change first channel to all zeros
    mask[:, :, 0] = 0

    # Use the mask to insert missingness in the data
    out = X.clone()
    out[mask] = np.nan

    # Concatenate the cumulative mask channel to the data
    out = torch.cat((out, cum_mask), dim=-1)

    # Fill forward where time is zero
    time_mask = time_mask.unsqueeze(-1).repeat(1, 1, X.shape[-1]+1)
    out[time_mask] = 0

    return out


def preprocess_for_transformer(X):
    """
    Preprocesses the output of insert_random_missingness for a transformer model.

    Parameters
    ----------
    X : torch.Tensor
        The input data with missingness inserted as NaNs across the datapoint.
        A cumulative mask channel is concatenated to the input data to indicate
        missingness. Shape (n_samples, sequence_length, n_features + 1)

    Returns
    -------
    input_ids : torch.Tensor
        The input tensor for the transformer model with special tokens for NaN values.
        Shape (n_samples, sequence_length, n_features + 1)
    attention_mask : torch.Tensor
        The attention mask tensor indicating the padding positions.
        Shape (n_samples, sequence_length)
    """
    # Convert NaN values to a special token
    input_ids = torch.nan_to_num(X, nan=0)  # Replace NaN with 0

    # Create attention mask
    attention_mask = ~torch.isnan(X[:, :, 0])  # Create mask based on the first feature (time)
    
    return input_ids, attention_mask


def fill_forward(X):
    '''
    Fill forward missing values in the data.

    Parameters
    ----------
    X : torch.Tensor
        The input data. Shape (n_samples, sequence_length, n_features)

    Returns
    -------
    out : torch.Tensor
        The input data with missing values filled in by the previous time point.
    '''
    # Create a mask that zeros out where time is zero
    time_mask = X[:, :, 0] == 0
    time_mask[:, 0] = 0

    out = X.clone()

    # Fill forward last observation
    for i in range(X.shape[0]):
        for j in range(1, X.shape[1]):
            if time_mask[i, j]:
                out[i, j] = out[i, j-1]

    return out