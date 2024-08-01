import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Load the input features
data = pd.read_csv('../../Data/preprocessedData/511_2024_04_03_training_data_v3_comma.csv')

# Define the target variable (y) and the features (X)
target = 'Tour Length [min]'
features = data.columns.difference([target])
# Replace -1 with NaN
data = data[~data[target].isin([-1, -2])]

data.replace([-1, -2], np.nan, inplace=True)

X = data[features]
y = data[target]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# Simplify the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, error_score='raise')
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the model on the training set with the best parameters
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Validate the model on the validation set
y_valid_pred = best_rf.predict(X_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
valid_mae = mean_absolute_error(y_valid, y_valid_pred)
valid_r2 = r2_score(y_valid, y_valid_pred)
print(f'Validation MSE: {valid_mse}')
print(f'Validation MAE: {valid_mae}')
print(f'Validation R2: {valid_r2}')

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='r2')
print(f"Cross-validated R2 scores: {cv_scores}")
print(f"Mean cross-validated R2 score: {cv_scores.mean()}")

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_valid, y_valid_pred, alpha=0.3, color='blue', label='Predicted')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual Tour Length')
plt.ylabel('Predicted Tour Length')
plt.title('Actual vs Predicted Tour Length')
plt.legend()
plt.grid(True)
plt.show()
