"""
Tune hyperparameters including interpolation method (linear, polynomial, pchip, etc), model hyperparams
(CNN stride, kernel, pooling, etc), LR, optimizer, etc)
Using time-series CV from time_series_cv.py

Usage:
    python tune_hyperparams.py --tune_method {exhaustive,randomized} [--n_iter 100]
"""

# imports
import pandas as pd
import numpy as np
import torch
import random
import argparse
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

def objective(params, feature_df, target_df, n_splits=3, epochs=10):
    """
    Trains a CNN model with given hyperparameters and evaluates it using time series cross-validation.
    Interpolates missing values in feature and target data using specified method.
    Returns a dict with params and evaluation metrics: avg_sharpe_ratio, std_sharpe_ratio, avg_mse, avg_mae, avg_r2
    """

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

    return result
    
    
    

def tune_exhaustive(feature_df, target_df, param_grid, n_splits=3, epochs=10):
    """
    Perform exhaustive hyperparameter tuning using time series cross-validation.
    
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

    def generate_hyperparameter_combinations(param_grid):
        """Generate all possible combinations of hyperparameters"""
        keys = param_grid.keys()
        values = param_grid.values()
        for combination in product(*values):
            yield dict(zip(keys, combination))

    #get total number of combinations:
    total_combinations = len(list(generate_hyperparameter_combinations(param_grid)))
    
    for params in tqdm(generate_hyperparameter_combinations(param_grid), total=total_combinations):
        print(f"\nTraining with params: {params}")
        
        result = objective(params, feature_df, target_df, n_splits=n_splits, epochs=epochs)
        results.append(result)
    
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/hyperparam_results_{timestamp}.csv', index=False)

    return pd.DataFrame(results)
                

def tune_randomized(feature_df, target_df, param_grid, n_iter=100, n_splits=3, epochs=10):
    """
    Perform randomized hyperparameter tuning using time series cross-validation.
    
    Args:
        feature_df: DataFrame containing features
        target_df: DataFrame containing targets
        param_grid: Dictionary of hyperparameters to search
        n_iter: Number of random parameter combinations to try
        n_splits: Number of time series CV splits
        epochs: Number of training epochs
        
    Returns:
        DataFrame with results for sampled hyperparameter combinations
    """
    results = []
    
    # Generate n_iter random parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = []
    
    # Ensure we don't try to sample more combinations than exist
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    n_iter = min(n_iter, total_combinations)
    
    # Generate unique random combinations
    seen = set()
    while len(seen) < n_iter:
        # Create a random combination
        random_combination = []
        for v in values:
            random_combination.append(random.choice(v))
        
        # Convert to tuple for hashing
        combo_tuple = tuple(random_combination)
        
        # Add if not already seen
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            param_combinations.append(dict(zip(keys, random_combination)))
    
    # Evaluate each random combination
    for params in tqdm(param_combinations, total=n_iter):
        print(f"\nTraining with params: {params}")
        
        result = objective(params, feature_df, target_df, n_splits=n_splits, epochs=epochs)
        results.append(result)
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/random_hyperparam_results_{timestamp}.csv', index=False)
    
    return pd.DataFrame(results)


def tune_bayesian(param_grid, feature_df, target_df, n_splits=3, epochs=10):
    """Tunes hyperparameters using Bayesian Optimization with Optuna library
    """
    import optuna

    def objective_optuna(trial):
        params = {}
        for k, v in param_grid.items():
            if isinstance(v[0], int):
                # For integers, use the first value as min and second as max
                params[k] = trial.suggest_int(k, min(v), max(v))
            elif isinstance(v[0], float):
                # For floats, use the first value as min and second as max
                params[k] = trial.suggest_float(k, min(v), max(v))
            elif isinstance(v[0], str):
                # For strings, pass the list directly
                params[k] = trial.suggest_categorical(k, v)
            else:
                raise TypeError(f'Unsupported hyperparameter type: {type(v[0])}')

        print(f"\nTraining with params: {params}")
        result = objective(params, feature_df, target_df, n_splits=n_splits, epochs=epochs)
        return result['avg_sharpe_ratio'] #maximize sharpe ratio


    study = optuna.create_study(direction='maximize')
    study.optimize(objective_optuna, n_trials=100)

    return study.best_params

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for CNN model')
    parser.add_argument('--tune_method', type=str, choices=['exhaustive', 'randomized', 'bayesian'], 
                       default='exhaustive',
                       help='Hyperparameter tuning method: exhaustive or randomized search')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Number of iterations for randomized search (default: 100)')
    args = parser.parse_args()
    
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
    
    
    # Run selected hyperparameter tuning method
    if args.tune_method == 'exhaustive':
        print(f"Running exhaustive hyperparameter search...")
        results = tune_exhaustive(
            feature_df=feature_df,
            target_df=target_df,
            param_grid=param_grid,
            n_splits=3,
            epochs=10
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(f'results/final_exhaustive_hyperparam_results_{timestamp}.csv', index=False)
    elif args.tune_method == 'randomized':
        print(f"Running randomized hyperparameter search with {args.n_iter} iterations...")
        results = tune_randomized(
            feature_df=feature_df,
            target_df=target_df,
            param_grid=param_grid,
            n_iter=args.n_iter,
            n_splits=3,
            epochs=10
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(f'results/final_random_hyperparam_results_{timestamp}.csv', index=False)

    elif args.tune_method == 'bayesian':
        print(f"Running Bayesian hyperparameter search...")
        results = tune_bayesian(
            feature_df=feature_df,
            target_df=target_df,
            param_grid=param_grid,
            n_splits=3,
            epochs=10
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(f'results/final_bayesian_hyperparam_results_{timestamp}.csv', index=False)

    
    # Print best parameters by Sharpe ratio
    best_idx = results['avg_sharpe_ratio'].idxmax()
    best_params = results.loc[best_idx].to_dict()
    
    print("\nBest parameters found:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
        
    # Save best parameters to a JSON file
    best_params_file = f'results/best_params_{args.tune_method}_{timestamp}.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\nBest parameters saved to {best_params_file}")
    best_params = results.loc[results['avg_sharpe_ratio'].idxmax()]
    print("\nBest parameters found:")
    print(best_params)

if __name__ == '__main__':
    main()