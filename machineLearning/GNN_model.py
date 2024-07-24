import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load data
features_df = pd.read_csv('../Data/Results/Training_input_features.csv')
test_data_df = pd.read_csv('../Data/Results/Testing_input_features.csv')

# Define the target variable (y) and the features (X)
target = 'tour_length'
features = features_df.columns.difference([target])

X = features_df[features].values
y = features_df[target].values

# Check for NaN values in the dataset
if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    print("Data contains NaN values. Please clean the dataset.")
    exit()

# Normalize features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1  # Handle zero std deviation by setting it to 1
X = (X - X_mean) / X_std

# Normalize target variable
y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare edge index (assuming a complete graph for simplicity)
def create_edge_index(num_nodes):
    return np.array([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]).T

# Convert to PyTorch tensors
x_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
edge_index_train = torch.tensor(create_edge_index(len(X_train)), dtype=torch.long)

x_valid = torch.tensor(X_valid, dtype=torch.float)
y_valid = torch.tensor(y_valid, dtype=torch.float)
edge_index_valid = torch.tensor(create_edge_index(len(X_valid)), dtype=torch.long)

# Create the data objects
train_data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
valid_data = Data(x=x_valid, edge_index=edge_index_valid, y=y_valid)

# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(X_train.shape[1], 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, 16)
        self.conv5 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        return x

# Initialize the model, loss function, and optimizer
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

# Implement gradient clipping
def train(model, data, optimizer, criterion, clip_value=1.0):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out.squeeze(), data.y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    return loss.item()

# Train the model
num_epochs = 500
best_val_loss = float('inf')
early_stopping_patience = 50
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train(model, train_data, optimizer, criterion)

    model.eval()
    with torch.no_grad():
        val_out = model(valid_data)
        val_loss = criterion(val_out.squeeze(), valid_data.y)

    print(f'Epoch {epoch}, Loss: {train_loss}, Validation Loss: {val_loss.item()}')

    # Early stopping
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping due to no improvement in validation loss")
        break

# Evaluate the model on the validation set
model.eval()
with torch.no_grad():
    pred_valid = model(valid_data).squeeze()
    pred_valid_np = pred_valid.numpy()

# Debugging: Check for NaN values in the predictions
if np.any(np.isnan(pred_valid_np)):
    print("Validation predictions contain NaN values. Please check the model output.")
    print("Invalid predictions:", pred_valid_np[np.isnan(pred_valid_np)])
    print("Validation features:", X_valid[np.isnan(pred_valid_np), :])

# Filter out NaN values before calculating metrics
mask = ~np.isnan(pred_valid_np)
if np.sum(mask) == 0:
    print("All validation predictions are NaN. Cannot compute metrics.")
else:
    # Denormalize the predictions
    pred_valid_np = pred_valid_np * y_std + y_mean
    y_valid_np = y_valid.numpy() * y_std + y_mean

    mse_valid = mean_squared_error(y_valid_np[mask], pred_valid_np[mask])
    r2_valid = r2_score(y_valid_np[mask], pred_valid_np[mask])
    rmse_valid = np.sqrt(mse_valid)

    print(f'Validation Mean Squared Error: {mse_valid}')
    print(f'Validation R-squared: {r2_valid}')
    print(f'Validation Root Mean Squared Error: {rmse_valid}')

    # Plot predicted vs actual values for the validation set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_valid_np[mask], pred_valid_np[mask], alpha=0.3, label='Predicted', color='r')
    plt.scatter(y_valid_np[mask], y_valid_np[mask], alpha=0.3, label='Actual', color='b')
    plt.plot([y_valid_np.min(), y_valid_np.max()], [y_valid_np.min(), y_valid_np.max()], 'k--', lw=2)
    plt.xlabel('Actual Tour Length')
    plt.ylabel('Predicted Tour Length')
    plt.title('GNN: Predicted vs. Actual Tour Length [Validation Set]')
    plt.legend()
    plt.show()

    # Prepare test data
    X_test = test_data_df[features].values
    X_test = (X_test - X_mean) / X_std
    x_test = torch.tensor(X_test, dtype=torch.float)
    edge_index_test = torch.tensor(create_edge_index(len(X_test)), dtype=torch.long)
    test_data = Data(x=x_test, edge_index=edge_index_test)

    # Predict on the test set
    model.eval()
    with torch.no_grad():
        y_test_pred = model(test_data).squeeze()

    # Check for NaN values in the test predictions
    y_test_pred_np = y_test_pred.numpy()
    if np.any(np.isnan(y_test_pred_np)):
        print("Test predictions contain NaN values. Please check the model output.")
        print("Invalid test predictions:", y_test_pred_np[np.isnan(y_test_pred_np)])
        print("Test features:", X_test[np.isnan(y_test_pred_np), :])

    # Denormalize the test predictions
    y_test_pred_np = y_test_pred_np * y_std + y_mean

    # Add the predictions to the test data
    test_data_df['Predicted Tour length'] = y_test_pred_np

    # Save the test data with predictions
    test_data_df.to_csv('../Data/Results/test_tour_predictions_gnn.csv', index=False)
    print("Predictions saved to '../Data/Results/test_tour_predictions_gnn.csv'")

    # Plot predicted vs actual values for the test set
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(range(len(y_test_pred_np))), y=y_test_pred_np, mode='markers', name='GNN Predictions',
                   marker=dict(color='red', opacity=0.3)))
    fig.update_layout(title='GNN Predictions [Test Set]', xaxis_title='Sample Index',
                      yaxis_title='Predicted Tour Length')
    fig.write_html('../Data/Graphs/gnn_test_predictions.html')
    fig.show()
