import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from model import CNN
from losses import SharpeLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, epochs, batch_size, X, y, criterion='sharpe', optimizer='adam', lr=0.001):
    """train model on given dataset
    """
    if criterion == 'sharpe':
        criterion = SharpeLoss()
    elif criterion == 'MSE':
        criterion = nn.MSELoss()
    elif criterion == 'MAE':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f'Invalid criterion: {criterion}')
    
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}')

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(device)
            y_batch = y[i:i+batch_size].to(device)

            optimizer.zero_grad()
           
            preds = model(X_batch) 
            loss = criterion(preds, y_batch.float())
            loss.backward()
            optimizer.step()

    return model

# eval functions copied from Kaggle comp:
SOLUTION_NULL_FILLER = -999999


def rank_correlation_sharpe_ratio(merged_df: pd.DataFrame) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).

    :param merged_df: DataFrame containing prediction columns (starting with 'prediction_')
                      and target columns (starting with 'target_')
    :return: Sharpe ratio of the rank correlation
    :raises ZeroDivisionError: If the standard deviation is zero
    """
    prediction_cols = [col for col in merged_df.columns if col.startswith('prediction_')]
    target_cols = [col for col in merged_df.columns if col.startswith('target_')]

    def _compute_rank_correlation(row):
        non_null_targets = [col for col in target_cols if not pd.isnull(row[col])]
        matching_predictions = [col for col in prediction_cols if col.replace('prediction', 'target') in non_null_targets]
        if not non_null_targets:
            raise ValueError('No non-null target values found')
        if row[non_null_targets].std(ddof=0) == 0 or row[matching_predictions].std(ddof=0) == 0:
            raise ZeroDivisionError('Denominator is zero, unable to compute rank correlation.')
        return np.corrcoef(row[matching_predictions].rank(method='average'), row[non_null_targets].rank(method='average'))[0, 1]

    daily_rank_corrs = merged_df.apply(_compute_rank_correlation, axis=1)
    std_dev = daily_rank_corrs.std(ddof=0)
    if std_dev == 0:
        raise ZeroDivisionError('Denominator is zero, unable to compute Sharpe ratio.')
    sharpe_ratio = daily_rank_corrs.mean() / std_dev
    return float(sharpe_ratio)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert all(solution.columns == submission.columns)

    submission = submission.rename(columns={col: col.replace('target_', 'prediction_') for col in submission.columns})

    # Not all securities trade on all dates, but solution files cannot contain nulls.
    # The filler value allows us to handle trading halts, holidays, & delistings.
    solution = solution.replace(SOLUTION_NULL_FILLER, None)
    return rank_correlation_sharpe_ratio(pd.concat([solution, submission], axis='columns'))
    


def eval(model, X_test: torch.tensor, y_true, targets: pd.DataFrame, batch_size=64):
    """evaluate model on MSE, MAE, R^2, and Kaggle eval metric:

    "The competition's metric is a variant of the Sharpe ratio, computed 
    by dividing the mean Spearman rank correlation between the predictions 
    and targets by the standard deviation"
    """


    model.eval()
    with torch.no_grad():
        y_pred = []
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            y_pred.extend(model(X_batch).cpu().numpy())

    y_pred = np.vstack(y_pred)
    y_true = y_true.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    target_ids = [col.split('_')[1] for col in targets.columns if col.startswith('target_')]
    pred_df_cols = [f'prediction_{id}' for id in target_ids]
    target_df_cols = [f'target_{id}' for id in target_ids]

    pred_df = pd.DataFrame(y_pred, columns=pred_df_cols)
    true_df = pd.DataFrame(y_true, columns=target_df_cols)

    merged_df = pd.concat([true_df, pred_df], axis=1)

    sharpe_ratio = rank_correlation_sharpe_ratio(merged_df)

    return mse, mae, r2, sharpe_ratio


#for getting train data
def get_X_y_overlap(feature_df: pd.DataFrame, target_df: pd.DataFrame, seq_length: int):
    """Gets features and targets for given dataframes
    input dfs should be split to fold
    contains overlap; each training example has seq_length-1 overlapping features; shouldn't be an issue but keep this in mind
    """
    X, y = [], []
    for i in range(len(feature_df)-seq_length):
        X.append(feature_df.iloc[i:i+seq_length].values)
        y.append(target_df.iloc[i+seq_length].values)
    return torch.tensor(X), torch.tensor(y)
    


def time_series_cv(model: nn.Module, feature_df: pd.DataFrame, target_df: pd.DataFrame, n_splits=3, epochs=10, batch_size=64, time_length=30, overlap=True, lr=0.001, criterion='MSE', optimizer='adam'):
    """perform time series cross validation"""

    n_samples = len(feature_df)
    fold_size = n_samples // n_splits 

    mses, maes, r2s, sharpe_ratios = [], [], [], []
    for i in range(1, n_splits+1):
        split_point = fold_size * i
        if len(feature_df[split_point:]) < time_length:
            print(f"Skipping fold {i} because the test set is too small.")
            continue

        X_train, y_train = get_X_y_overlap(feature_df[:split_point].drop(columns=['date_id']), target_df[:split_point].drop(columns=['date_id']), time_length)
        X_test, y_test = get_X_y_overlap(feature_df[split_point:].drop(columns=['date_id']), target_df[split_point:].drop(columns=['date_id']), time_length)

        
        model = train(model, epochs, batch_size, X_train, y_train, lr=lr, criterion=criterion, optimizer=optimizer)

        mse, mae, r2, sharpe_ratio = eval(model, X_test, y_test, target_df[split_point:])

        mses.append(mse)
        maes.append(mae)
        r2s.append(r2)
        sharpe_ratios.append(sharpe_ratio)

    return {'mses': mses, 'maes': maes, 'r2s': r2s, 'sharpe_ratios': sharpe_ratios}



def test():

    train_df = pd.read_csv('../data/train.csv')
    target_df = pd.read_csv('../data/train_labels.csv')
    

    model = CNN(num_features=len(train_df.columns)-1, time_length=30, num_targets=len(target_df.columns)-1)

    #deal with nan values
    train_df.fillna(0, inplace=True)
    target_df.fillna(0, inplace=True)

    metrics = time_series_cv(model, train_df, target_df, time_length=30, epochs=1, criterion='MSE')
    print(f"Mean MSE: {np.mean(metrics['mses'])}")
    print(f"Mean MAE: {np.mean(metrics['maes'])}")
    print(f"Mean R2: {np.mean(metrics['r2s'])}")
    print(f"Mean Sharpe Ratio: {np.mean(metrics['sharpe_ratios'])}")


    
