# Mitsui & Co. Commodity Prediction Challenge

This repository contains my submission for the [Mitsui & Co. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/overview) on Kaggle. The goal of this competition is to predict the future prices of various commodities.

## Project Structure

This repository includes my two main approaches:

-   `linear_models.ipynb`: A Jupyter notebook exploring baseline linear models, such as Ridge Regression. This serves as a starting point to understand the data and establish a performance baseline.
-   `cnn/`: A more sophisticated approach using a 1D Convolutional Neural Network (CNN) implemented in PyTorch to capture temporal patterns in the time series data.

## Quick Start

1.  **Download the data:**

    Register & download the competition data from [this link](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data) and place it in a directory named `data/` in the root of this project.

2.  **Set up the environment:**

    ```bash
    conda env create -f env.yml
    conda activate mitsui
    ```

3.  **Run hyperparameter tuning:**

    ```bash
    python cnn/tune_hyperparams.py
    ```

    The results of the hyperparameter tuning will be saved as a CSV file in the `cnn/results/` directory.

## Methodology

### Linear Models

The `linear_models.ipynb` notebook contains my initial exploration of the data. I implemented a simple Ridge Regression model to predict future returns. This involved:

-   Basic feature engineering and data cleaning.
-   Implementing a time-series cross-validation scheme to evaluate the model's performance without data leakage.
-   Establishing a baseline to compare more complex models against.

### Convolutional Neural Network (CNN)

The `cnn/` directory contains a more advanced model using a 1D CNN. This approach is designed to learn hierarchical features from the time series data. The key components are:

-   **`model.py`**: Defines the CNN architecture, which includes multiple convolutional layers, batch normalization, pooling, and dropout for regularization.
-   **`time_series_cv.py`**: Implements a robust time-series cross-validation framework for training and evaluating the CNN. This script uses a custom evaluation metric based on the Sharpe ratio of rank correlations, as specified by the competition.
-   **`tune_hyperparams.py`**: This script is used for hyperparameter tuning of the CNN. It leverages various interpolation methods, such as **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial), to efficiently search the hyperparameter space. PCHIP is commonly used in time-series financial data and preserves the shape of the data and avoids overshoots.

## Next Steps

Given more time, I would explore the following to improve the model's performance:

-   **Recurrent Neural Networks (RNNs)**: Implement more complex sequence models like LSTMs or GRUs, which are well-suited for time-series forecasting.
-   **Transformers**: Explore the use of Transformer architectures, which have shown state-of-the-art performance in many sequence modeling tasks.
-   **Feature Engineering**: Develop more sophisticated features, such as volatility measures, rolling statistics, and technical indicators.
-   **Ensemble Methods**: Combine the predictions of multiple models (e.g., linear models, CNNs, and RNNs) to create a more robust and accurate final prediction.
