import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import plotly.graph_objects as go

# Create the map folder if it doesn't exist
if not os.path.exists('../Data/Graphs'):
    os.makedirs('../Data/Graphs')

# Load training data
train_data = pd.read_csv('../Data/Results/Training_input_features.csv')

# Load test data (for final evaluation only)
test_data_rf = pd.read_csv('../Data/Results/Testing_input_features.csv')

# Replace -1 with NaN
train_data.replace(-1, pd.NA, inplace=True)
test_data_rf.replace(-1, pd.NA, inplace=True)

# Fill NaN values with the median of each column
train_data.fillna(train_data.median(), inplace=True)
test_data_rf.fillna(test_data_rf.median(), inplace=True)

# Define the target variable (y) and the features (X)
target = 'tour_length'
features = train_data.columns.difference([target])

X = train_data[features]
y = train_data[target]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Randomized Search with Cross-Validation
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", random_search.best_params_)

# Use the best model
best_model = random_search.best_estimator_

# Predict on the validation set
y_valid_pred = best_model.predict(X_valid)

# Calculate evaluation metrics
mse = mean_squared_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)
rmse = np.sqrt(mse)

print(f'Validation Mean Squared Error: {mse}')
print(f'Validation R-squared: {r2}')
print(f'Validation Root Mean Squared Error: {rmse}')

# Predict on the test set (assuming you have a test set)
X_test = test_data_rf[features]
y_test_pred = best_model.predict(X_test)

# Add the predictions to the test data
test_data_rf['Predicted Tour length[min]'] = y_test_pred

# Save the test data with predictions
test_data_rf.to_csv('../Data/Results/test_tour_predictions_rf.csv', index=False)
print("Predictions saved to '../Data/Results/test_tour_predictions_rf.csv'")


# Plot predicted vs actual values for the validation set using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_valid, y=y_valid_pred, mode='markers', name='Predicted', marker=dict(color='red', opacity=0.3)))
fig.add_trace(go.Scatter(x=y_valid, y=y_valid, mode='markers', name='Actual', marker=dict(color='blue', opacity=0.3)))
fig.add_trace(go.Scatter(x=[y_valid.min(), y_valid.max()], y=[y_valid.min(), y_valid.max()], mode='lines', name='Ideal', line=dict(color='black', dash='dash')))
fig.update_layout(title='Random Forest: Predicted vs. Actual Tour Length [Validation Set]', xaxis_title='Actual Tour Length', yaxis_title='Predicted Tour Length')
fig.write_html('../Data/Graphs/rf_validation.html')
fig.show()

