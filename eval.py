import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def _retrend(y_pred : pd.DataFrame, full_df : pd.DataFrame, feature : str, intervals : list[int] = []):
    """
    :y_pred: pd.DataFrame['ds', 'y'] : containing only predictions
    :full_df: pd.DataFrame['ds', 'y'] : containing full data
    :feature: string : target feature to transform
    :interval: list[int] : intervals to undo, should be empty if feature is 'y'

    Undoing log and differencing
    
    RETURN : y_pred with 'y' column unlogged and retrended
    """

    combined = pd.merge_asof(
        y_pred.sort_values('ds'), 
        full_df[['ds', 'y_log']].sort_values('ds'), 
        on='ds', 
        allow_exact_matches=False, 
        direction='backward'
    )
    combined.to_csv('combined.csv', index=False)
    combined[feature] = np.exp(combined['y_log'] + combined[feature])

    if combined[feature].isnull().any():
        raise ValueError("Could not find preceding values for one or more predictions.")

    y_pred[feature] = combined[feature]
    
    if feature != 'y':
        for interval in intervals:
            lo_name, hi_name = f"{feature}-lo-{interval}", f"{feature}-hi-{interval}" 
            combined[lo_name] = np.exp(combined['y_log'] + combined[lo_name])
            combined[hi_name] = np.exp(combined['y_log'] + combined[hi_name])
            y_pred[lo_name] = combined[lo_name]
            y_pred[hi_name] = combined[hi_name]

def _unlog(y_pred : pd.DataFrame, _, feature : str, intervals : list[int] = []):
    y_pred[feature] = np.exp(y_pred[feature])
    if feature != 'y':
        for interval in intervals:
            lo_name, hi_name = f"{feature}-lo-{interval}", f"{feature}-hi-{interval}" 
            y_pred[lo_name] = np.exp(y_pred[lo_name])
            y_pred[hi_name] = np.exp(y_pred[hi_name])


def _graph(pred_df, reals, preds, feature, mse, percent_error, transform, intervals):
    # Build interval bands (pairs of lo/hi columns)
    interval_bands = []
    for interval in intervals:
        lo_name = f"{feature}-lo-{interval}"
        hi_name = f"{feature}-hi-{interval}"
        if lo_name in pred_df.columns and hi_name in pred_df.columns:
            interval_bands.append((lo_name, hi_name, interval))
    
    plt.figure(figsize=(12, 6))
    
    # Define colors for different confidence intervals
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightsalmon']
    
    # Plot confidence bands if they exist
    if interval_bands:
        for idx, (lo_name, hi_name, interval) in enumerate(interval_bands):
            color = colors[idx % len(colors)]
            plt.fill_between(
                pred_df.index, 
                pred_df[lo_name], 
                pred_df[hi_name],
                alpha=0.3,
                color=color,
                label=f'{interval}% Confidence Interval'
            )
    
    # Plot actual and predicted lines
    plt.plot(pred_df.index, reals, label='Actual', alpha=0.7, linewidth=2)
    plt.plot(pred_df.index, preds, label='Predicted', alpha=0.7, linewidth=2)
    
    plt.title(f'{feature} - MSE: {mse:.2f}, MAPE: {percent_error:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./baseline/forecasts/fc_graphed{transform}.png')
    plt.close()


transforms = {
    # 'lightgbm' : (_unlog, 'LGBMRegressor', []),
    # 'mstla' : (_unlog, 'MSTL', [50, 80, 95]),
    # 'nhits' : (_retrend, 'NHITS', []),
    'sarimax' : (_retrend, 'AutoARIMA', [50, 80, 95]),
    # 'tbats' : (None, 'AutoTBATS', [50, 80, 95])
}
# mstla, sarimax, tbats
test_df = pd.read_csv('./dataset/cleaned/auth_order_test.csv')
train_df = pd.read_csv('./dataset/cleaned/auth_order_train.csv')

full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df['ds'] = pd.to_datetime(full_df['ds'])


for transform in transforms.keys():
    if transform is None:
        continue
    
    inverter = transforms[transform][0]
    feature = transforms[transform][1]
    intervals = transforms[transform][2]

    pred_df = pd.read_csv(f"./baseline/cv_{transform}.csv")

    pred_df['ds'] = pd.to_datetime(pred_df['ds'])

    if inverter is not None:
        inverter(pred_df, full_df, feature, intervals)
        inverter(pred_df, full_df, 'y')

    preds = pred_df[feature]
    reals = pred_df['y']

    mse = mean_squared_error(reals, preds)
    percent_error = np.mean(np.abs((reals - preds) / reals)) * 100
    _graph(pred_df, reals, preds, feature, mse, percent_error, transform, intervals)



