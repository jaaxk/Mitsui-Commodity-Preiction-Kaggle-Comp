"""
Tune hyperparameters including interpolation method (linear, polynomial, pchip, etc), model hyperparams
(CNN stride, kernel, pooling, etc), LR, optimizer, etc)
Using time-series CV from time_series_cv.py
"""


# imports
import pandas as pd
import numpy as np
import torch
from model import CNN
from time_series_cv import time_series_cv
from itertools import product
import json
from datetime import datetime
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator
import torch

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def generate_hyperparameter_combinations(param_grid):
    """Generate all possible combinations of hyperparameters"""
    keys = param_grid.keys()
    values = param_grid.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))


def pchip_interpolate(df):
    """
    Apply PCHIP interpolation along each column of a DataFrame.
    Handles NaNs by first forward/backward filling.
    """
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Forward/backward fill to handle leading/trailing NaNs
    result = result.ffill().bfill()
    
    # Get indices of non-NaN values
    x = np.arange(len(df))
    valid_mask = ~df.isna().any(axis=1)
    
    # Only proceed if we have at least 2 valid points
    if valid_mask.sum() >= 2:
        x_valid = x[valid_mask]
        
        # Apply PCHIP to each column
        for col in df.columns:
            y_valid = df[col].values[valid_mask]
            pchip = PchipInterpolator(x_valid, y_valid)
            result[col] = pchip(x)
    
    return result

def tune_hyperparameters(feature_df, target_df, param_grid, n_splits=3, epochs=10):
    """
    Perform hyperparameter tuning using time series cross-validation.
    
    Args:
        feature_df: DataFrame containing features
        target_df: DataFrame containing targets
        param_grid: Dictionary of hyperparameters to search
        n_splits: Number of time series CV splits
        epochs: Number of training epochs
        
    Returns:
        DataFrame with results for all hyperparameter combinations
    """
    results = []
    
    for params in tqdm(generate_hyperparameter_combinations(param_grid)):
        print(f"\nTraining with params: {params}")
        
        # Initialize model with current hyperparameters
        model_params = {
            'num_features': len(feature_df.columns)-1,
            'time_length': params.get('time_length', 30),
            'num_targets': len(target_df.columns)-1,
            'conv_1_out': params.get('conv_1_out', 64),
            'conv_2_out': params.get('conv_2_out', 128),
            'conv_1_kernel_size': params.get('conv_1_kernel_size', 3),
            'conv_2_kernel_size': params.get('conv_2_kernel_size', 3),
            'pooling': params.get('pooling', 'max'),
            'pool_1_kernel_size': params.get('pool_1_kernel_size', 2),
            'pool_2_kernel_size': params.get('pool_2_kernel_size', 2),
            'hidden_layer_size': params.get('hidden_layer_size', 256),
            'dropout': params.get('dropout', 0.3)
        }

        
        model = CNN(**model_params).to(device)

        if params.get('interpolation_method') == 'pchip':
            # Handle missing values with PCHIP interpolation
            feature_df_interpolated = pchip_interpolate(feature_df)
            target_df_interpolated = pchip_interpolate(target_df)


        elif params.get('interpolation_method') == 'polynomial':
            # Handle missing values with polynomial interpolation
            interp_order = params.get('interpolation_order', 1)  # 1=linear, 2=quadratic, etc.
            
            # Apply polynomial interpolation with specified order
            feature_df_interpolated = feature_df.interpolate(
                method='polynomial', 
                order=interp_order
            )
            target_df_interpolated = target_df.interpolate(
                method='polynomial',
                order=interp_order
            )
        else:
            raise ValueError(f'Invalid interpolation method: {params.get("interpolation_method")}')
        
        # Handle any remaining NAs at the edges
        feature_df_interpolated = feature_df_interpolated.ffill().bfill()
        target_df_interpolated = target_df_interpolated.ffill().bfill()
        


        # Train and evaluate using time series CV
        metrics = time_series_cv(
            model=model,
            feature_df=feature_df_interpolated,
            target_df=target_df_interpolated,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=params.get('batch_size', 64),
            lr=params.get('lr', 0.001),
            criterion=params.get('criterion', 'MSE'),
            optimizer=params.get('optimizer', 'adam')
        )
        
        # Store results
        result = {
            **params,
            'avg_sharpe_ratio': np.mean(metrics['sharpe_ratios']),
            'std_sharpe_ratio': np.std(metrics['sharpe_ratios']),
            'avg_mse': np.mean(metrics['mses']),
            'avg_mae': np.mean(metrics['maes']),
            'avg_r2': np.mean(metrics['r2s'])
        }
        
        results.append(result)
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/hyperparam_results_{timestamp}.csv', index=False)
    
    return pd.DataFrame(results)

def main():
    # Load your data
    feature_df = pd.read_csv('../data/train.csv')
    target_df = pd.read_csv('../data/train_labels.csv')
    
    # Define hyperparameter search space
    param_grid = {
        'conv_1_out': [32, 64, 128],
        'conv_2_out': [64, 128, 256],
        'conv_1_kernel_size': [3, 5, 7],
        'conv_2_kernel_size': [3, 5, 7],
        'pooling': ['max', 'avg'],
        'pool_1_kernel_size': [2, 3],
        'pool_2_kernel_size': [2, 3],
        'hidden_layer_size': [128, 256, 512],
        'dropout': [0.2, 0.3, 0.4],
        'batch_size': [32, 64, 128],
        'lr': [0.001, 0.0001, 0.00001],
        'criterion': ['MSE', 'MAE'],
        'optimizer': ['adam', 'sgd'],
        'interpolation_order': [1, 2, 3],  # 1=linear, 2=quadratic
        'interpolation_method': ['pchip', 'polynomial']
    }
    
    # Run hyperparameter tuning
    results = tune_hyperparameters(
        feature_df=feature_df,
        target_df=target_df,
        param_grid=param_grid,
        n_splits=3,
        epochs=10
    )
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(f'results/final_hyperparam_results_{timestamp}.csv', index=False)
    
    # Print best parameters
    best_params = results.loc[results['avg_sharpe_ratio'].idxmax()]
    print("\nBest parameters found:")
    print(best_params)

if __name__ == '__main__':
    main()