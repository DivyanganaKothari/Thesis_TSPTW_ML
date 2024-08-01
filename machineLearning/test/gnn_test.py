import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load the input features
data = pd.read_csv('../../Data/TestInputFeaturesCheck6/input_features_test_1.csv')

# Define the target variable (y) and the features (X)
target = 'Tour Length'
features = data.columns.difference([target])



# Replace -1 with NaN and drop rows with NaN values
data.replace([-1], np.nan, inplace=True)

# Separate the features and target variable
X = data[features].values
y = data[target].values
# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create edge index (example: fully connected graph)
num_nodes = X.shape[0]
edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes)], dtype=torch.long).t().contiguous()

# Create PyTorch Geometric data object
graph_data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y, dtype=torch.float).view(-1, 1))
# Split indices into training and validation sets
indices = np.arange(num_nodes)
train_indices, val_indices = train_test_split(indices, test_size=0.25, random_state=42)

# Create masks for training and validation nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask[val_indices] = True

# Assign masks to the data object
graph_data.train_mask = train_mask
graph_data.val_mask = val_mask


class GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x
# Initialize the model, loss function, and optimizer
model = GNN(num_node_features=X.shape[1])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = model(graph_data)[graph_data.train_mask]
    loss = criterion(output, graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Validate the model
model.eval()
with torch.no_grad():
    val_output = model(graph_data)[graph_data.val_mask]
    val_loss = criterion(val_output, graph_data.y[graph_data.val_mask])
    val_r2 = r2_score(graph_data.y[graph_data.val_mask].numpy(), val_output.numpy())
    print(f'Validation Loss: {val_loss.item()}')
    print(f'Validation R2: {val_r2}')

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(graph_data.y[graph_data.val_mask].numpy(), val_output.numpy(), alpha=0.3, color='blue', label='Predicted')
plt.plot([graph_data.y[graph_data.val_mask].min(), graph_data.y[graph_data.val_mask].max()], [graph_data.y[graph_data.val_mask].min(), graph_data.y[graph_data.val_mask].max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual Tour Length')
plt.ylabel('Predicted Tour Length')
plt.title('Actual vs Predicted Tour Length')
plt.legend()
plt.grid(True)
plt.show()
