#try cnn as well
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
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


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_valid_scaled, y_valid), verbose=2)

# Validate the model on the validation set
y_valid_pred = model.predict(X_valid_scaled).flatten()
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
