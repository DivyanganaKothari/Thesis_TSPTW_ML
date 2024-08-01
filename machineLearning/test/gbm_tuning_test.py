import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the input features
data = pd.read_csv('../../Data/TestInputFeaturesCheck4/input_features_test_1.csv')

# Define the target variable (y) and the features (X)
target = 'Tour Length'
features = data.columns.difference([target])

X = data[features]
y = data[target]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the GradientBoostingRegressor
gbm = GradientBoostingRegressor(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the model on the training set with the best parameters
best_gbm = GradientBoostingRegressor(**best_params, random_state=42)
best_gbm.fit(X_train, y_train)

# Validate the model on the validation set
y_valid_pred = best_gbm.predict(X_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
valid_mae = mean_absolute_error(y_valid, y_valid_pred)
valid_r2 = r2_score(y_valid, y_valid_pred)
print(f'Validation MSE: {valid_mse}')
print(f'Validation MAE: {valid_mae}')
print(f'Validation R2: {valid_r2}')

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(best_gbm, X, y, cv=5, scoring='r2')
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
