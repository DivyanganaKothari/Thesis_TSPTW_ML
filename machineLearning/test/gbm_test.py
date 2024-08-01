import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

# Load the input features
data = pd.read_csv('../../Data/TestInputFeaturesCheck7/input_features_test_1.csv')

# Define the target variable (y) and the features (X)
target = 'Tour Length'
features = data.columns.difference([target])

data.replace([-1], np.nan, inplace=True)


X = data[features]
y = data[target]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the GradientBoostingRegressor
gbm = GradientBoostingRegressor(random_state=42)

# Train the model on the training set
gbm.fit(X_train, y_train)

# Validate the model on the validation set
y_valid_pred = gbm.predict(X_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
valid_r2 = r2_score(y_valid, y_valid_pred)
print(f'Validation MSE: {valid_mse}')
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
