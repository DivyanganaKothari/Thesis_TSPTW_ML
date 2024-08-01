import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the input features
data = pd.read_csv('../../Data/TestInputFeaturesCheck7/input_features_test_1.csv')

# Define the target variable (y) and the features (X)
target = 'Tour Length'
features = data.columns.difference([target])

# Replace -1 with NaN
#data = data[~data[target].isin(-1)]

data.replace([-1], np.nan, inplace=True)


# Define feature matrix X and target vector y
X = data[features]
y = data[target]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50, random_state=42)

# Train the model on the training set
rf.fit(X_train, y_train)

# Validate the model on the validation set
y_valid_pred = rf.predict(X_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
valid_rmse = np.sqrt(valid_mse)
valid_r2 = r2_score(y_valid, y_valid_pred)
print(f'Validation MSE: {valid_mse}')
print(f'Validation RMSE: {valid_rmse}')
print(f'Validation R2: {valid_r2}')

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
