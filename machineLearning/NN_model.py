import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go

# Create the map folder if it doesn't exist
if not os.path.exists('../Data/Graphs'):
    os.makedirs('../Data/Graphs')

# Load training data
train_data = pd.read_csv('../Data/Results/Training_input_features.csv')

# Load test data (for final evaluation only)
test_data_nn = pd.read_csv('../Data/Results/Testing_input_features.csv')

# Replace -1 with NaN
train_data.replace(-1, pd.NA, inplace=True)
test_data_nn.replace(-1, pd.NA, inplace=True)

# Fill NaN values with the median of each column
train_data.fillna(train_data.median(), inplace=True)
test_data_nn.fillna(test_data_nn.median(), inplace=True)

# Define the target variable (y) and the features (X)
target = 'tour_length'
features = train_data.columns.difference([target])

X_train = train_data[features]
y_train = train_data[target]

X_test_nn = test_data_nn[features]  # No target column in test data

# Split the training data into training and validation sets
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_split = scaler.fit_transform(X_train_split)
X_valid_split = scaler.transform(X_valid_split)
X_test_nn = scaler.transform(X_test_nn)

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_split.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Add dropout layer for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # Add dropout layer for regularization
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_split, y_train_split, validation_data=(X_valid_split, y_valid_split), epochs=100, batch_size=32, callbacks=[early_stopping])

# Predict on the validation set
y_valid_pred = model.predict(X_valid_split).flatten()

# Calculate evaluation metrics on the validation set
mse_valid = mean_squared_error(y_valid_split, y_valid_pred)
rmse_valid = np.sqrt(mse_valid)
mae_valid = mean_absolute_error(y_valid_split, y_valid_pred)
r2_valid = r2_score(y_valid_split, y_valid_pred)

print(f'Validation Mean Squared Error: {mse_valid}')
print(f'Validation Root Mean Squared Error: {rmse_valid}')
print(f'Validation Mean Absolute Error: {mae_valid}')
print(f'Validation R-squared: {r2_valid}')

# Plot predicted vs actual values for the validation set using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_valid_split, y=y_valid_pred, mode='markers', name='Predicted', marker=dict(color='red', opacity=0.3)))
fig.add_trace(go.Scatter(x=y_valid_split, y=y_valid_split, mode='markers', name='Actual', marker=dict(color='blue', opacity=0.3)))
fig.add_trace(go.Scatter(x=[y_valid_split.min(), y_valid_split.max()], y=[y_valid_split.min(), y_valid_split.max()], mode='lines', name='Ideal', line=dict(color='black', dash='dash')))
fig.update_layout(title='Neural Network: Predicted vs. Actual Tour Length [Validation Set]', xaxis_title='Actual Tour Length', yaxis_title='Predicted Tour Length')
fig.write_html('../Data/Graphs/nn_validation.html')
fig.show()

# Predict on the test set
y_test_pred_nn = model.predict(X_test_nn).flatten()

# Add the predictions to the test data
test_data_nn['Predicted Tour length[min]'] = y_test_pred_nn

# Save the test data with predictions
test_data_nn.to_csv('../Data/Results/test_tour_predictions_nn.csv', index=False)
print("Predictions saved to '../Data/Results/test_tour_predictions_nn.csv'")

# Load Random Forest predictions for comparison
rf_predictions = pd.read_csv('../Data/Results/test_tour_predictions_rf.csv')['Predicted Tour length[min]']

# Compare Random Forest and Neural Network predictions using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(y_test_pred_nn))), y=y_test_pred_nn, mode='markers', name='Neural Network', marker=dict(color='red', opacity=0.3)))
fig.add_trace(go.Scatter(x=list(range(len(rf_predictions))), y=rf_predictions, mode='markers', name='Random Forest', marker=dict(color='blue', opacity=0.3)))
fig.update_layout(title='Random Forest vs. Neural Network Predictions [Test Set]', xaxis_title='Sample Index', yaxis_title='Predicted Tour Length')
fig.write_html('../Data/Graphs/rf_vs_nn_test.html')
fig.show()
