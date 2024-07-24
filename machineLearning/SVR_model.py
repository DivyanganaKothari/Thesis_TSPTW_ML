import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

# Load the features data
features_df = pd.read_csv('../Data/Results/Training_input_features.csv')

# Define the target variable (y) and the features (X)
target = 'tour_length'
features = features_df.columns.difference([target])

X = features_df[features]
y = features_df[target]

# Ensure no missing values
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVR model and hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.1, 0.2, 0.5, 1, 2],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],  # Only used for 'poly' kernel
    'gamma': ['scale', 'auto']  # Only used for 'rbf', 'poly', and 'sigmoid' kernels
}

svr = SVR()
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Use the best model
best_model = grid_search.best_estimator_

# Predict on the validation set
y_valid_pred = best_model.predict(X_valid)

# Calculate evaluation metrics
mse = mean_squared_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Root Mean Squared Error: {rmse}')

# Plot predicted vs actual values for the validation set
plt.figure(figsize=(10, 6))
plt.scatter(y_valid, y_valid_pred, alpha=0.3, label='Predicted', color='r')
plt.scatter(y_valid, y_valid, alpha=0.3, label='Actual', color='b')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=2)
plt.xlabel('Actual Tour Length')
plt.ylabel('Predicted Tour Length')
plt.title('SVR: Predicted vs. Actual Tour Length [Validation Set]')
plt.legend()
plt.show()
