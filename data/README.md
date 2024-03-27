# Modelling Sequential Data with Neural Differential Equations

## Course: ECSE 552 - Deep Learning

### University: McGill University

#### Authors: Tanaka Akiyama, Gian Favero, Maxime Favreau-Vachon, Mohamed Mohamed

##### Date: April 12th, 2024

---

## Dataset Structure

### Character Trajectories - Classification

The training data is of shape (batch, seq_length, channels). Each sequence is padded with zeros to be the length of the largest sequence in the dataset. A channel for time is added, which spans from 0 to length-1. The data is normalized by channel for each sequence. The labels are the class of the sequence, which are integer values from 0 to 19 (20 classes total). The data is split into an 80-20 train and validation set, stratified based on the labels. 

The data pre-processing is only partially done, as each model needs specific formatting to handle irregular sampling (ie. deleting random parts of each sequence). For the torchcde based models, missing values need to be filled in with NaN and observation masks have to be appended as a channel - this is slightly different for RNN/LSTM models. In addition, the padding needs to be handled for the torchcde models using a fill forward rather than naive zero padding.

For irregular sampling experiments, it is recommended to create custom transform methods for each model. When importing the dataset a portion of the sequence will be masked with the corresponding channels added. This again will vary from model to model, so various transform methods (or cases) need to be accounted for.

### Character Trajectories - Regression

The training data is of shape (batch, seq_length, channels), where the first 2/3 of the original sequence is kept, and the rest is zero padding. An identical time channel is added and the same normalization method is applied. The labels are the remaining 1/3 of the original sequence, with the rest padded with zeros. 

To accommodate the data for each model, similar methods need to be enforced as in classification (fill forward for torchcde). Either the same custom transform methods can be applied again (with an option to not mask any values). 