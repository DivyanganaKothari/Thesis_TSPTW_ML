import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def mean_percentage_error(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#def evaluate_model(y_valid, y_valid_pred, feature_set_name, model_name):
def evaluate_model(y_true, y_pred, feature_set_name, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    k = len(y_pred)
    adj_r2 = adjusted_r2_score(r2, n, k)
    mpe = mean_percentage_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f'Validation Mean Squared Error for {model_name} ({feature_set_name}): {mse}')
    print(f'Validation R-squared for {model_name} ({feature_set_name}): {r2}')
    print(f'Validation Root Mean Squared Error for {model_name} ({feature_set_name}): {rmse}')

    return {
        "model": model_name,
        "feature_set": feature_set_name,
        "mse": mse,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "mpe": mpe,
        "mape": mape
        #"y_valid_pred": y_valid_pred,
        #"y_valid": y_valid
    }
"""
#plot the results of the model

def plot_results(y_valid, y_valid_pred, feature_set_name, model_name, graphs_dir):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=y_valid, y=y_valid_pred, mode='markers', name='Predicted', marker=dict(color='red', opacity=0.3)))
    fig.add_trace(
        go.Scatter(x=y_valid, y=y_valid, mode='markers', name='Actual', marker=dict(color='blue', opacity=0.3)))

    fig.update_layout(
        title=f'{model_name}: Predicted vs. Actual Tour Length [Validation Set] ({feature_set_name})',
        xaxis_title='Actual Tour Length',
        yaxis_title='Predicted Tour Length',
        showlegend=True
    )

    graph_path = os.path.join(graphs_dir, f'predicted_vs_actual_validation_set_{model_name}_{feature_set_name}.html')
    fig.write_html(graph_path)
    print(f"Graph saved to {graph_path}")
    fig.show()
    """
