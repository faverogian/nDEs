# Modelling Sequential Data with Neural Differential Equations

## Course: ECSE 552 - Deep Learning

### University: McGill University

#### Authors: Tanaka Akiyama, Gian Favero, Maxime Favreau-Vachon, Mohamed Mohamed

##### Date: April 12th, 2024

---

In this study, we conduct a comparative analysis of sequential models for
time-series data, focusing on nDEs (both Neural Controlled Differential Equations (nCDE) and Neural Ordinary Differential Equations (nODE))
alongside modern discrete methods like LSTM networks, RNNs, and Transformers. Specifically, we compare their performance in tasks involving the classification of Latin characters (Encoder Task), forecasting of Latin characters (Sequence-to-Sequence Task), and forecasting of weather (Decoder Task). 

The nCDE model outperforms Transformer, LSTM, and RNN baselines in Latin character classification, with fewer parameters, while in sequence-to-sequence tasks, the Transformer model excels due to self-attention, and the nCDE-nODE model achieves comparable performance with fewer parameters, although self-attention-based models better capture intricate details. In the decoding task, the nODE model performs slightly below RNN and LSTM models in terms of MSE but uses significantly fewer parameters. 

Our investigation revealed nDE’s remarkable ability to handle irregularly sampled data with superior memory efficiency. However, their slower computation speed and limitations in real-time settings pose challenges. Nonetheless, nDEs offer continuous representations of time series, excel on irregularly sampled data, and exhibit memory efficiency, making them compelling options in resource-constrained scenarios.

The full paper can be seen [here](https://github.com/faverogian/nDEs/blob/main/report/report.pdf).

## Repository

This repository is structured as follows:

- **`data/`**: Contains raw and processed data used in the project.
  - `raw/`: Raw data files.
  - `processed/`: Processed data files.
- **`notebooks/`**: Jupyter notebooks for exploration, preprocessing, and modeling.
  - `exploratory/`: Notebooks for data exploration.
  - `preprocessing/`: Notebooks for data preprocessing.
  - `modeling/`: Notebooks for model training and evaluation.
- **`src/`**: Source code for the project.
  - `data/`: Scripts for data loading, preprocessing, and transformation.
  - `models/`: Scripts for defining and training ML models.
  - `evaluation/`: Scripts for model evaluation, metrics, and visualization.
  - `utils/`: Utility scripts and helper functions.
- **`experiments/`**: Directory for experiment scripts and logs.
- **`report/`**: Project figures and visualizations for reports and presentations.
- **`requirements.txt`**: Python dependencies file.
- **`README.md`**: Overview of the project, setup instructions, and usage guide.

## Installation

All required packages in ```requirements.txt``` can be installed via ```pip```.

## Contact

For questions or feedback, please contact [Gian Favero](https://faverogian.github.io/).
