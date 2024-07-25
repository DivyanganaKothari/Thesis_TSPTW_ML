# train_gnn.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from models_evaluate_plot import evaluate_model#, plot_results

class GNN(nn.Module):
    def __init__(self, input_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x

#def train_gnn(X_train, y_train, X_valid, y_valid, feature_set_name, model_dir):
def train_gnn(X_train, y_train, X_valid, y_valid, X_test, feature_set_name):
    def create_edge_index(num_nodes):
        return torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()

    x_train = torch.tensor(X_train.values, dtype=torch.float)
    y_train = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)
    edge_index_train = create_edge_index(len(X_train))

    x_valid = torch.tensor(X_valid.values, dtype=torch.float)
    y_valid = torch.tensor(y_valid.values, dtype=torch.float).view(-1, 1)
    edge_index_valid = create_edge_index(len(X_valid))

    x_test = torch.tensor(X_test.values, dtype=torch.float)
    edge_index_test = create_edge_index(len(X_test))

    input_dim = X_train.shape[1]

    train_data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
    valid_data = Data(x=x_valid, edge_index=edge_index_valid, y=y_valid)
    test_data = Data(x=x_test, edge_index=edge_index_test)


    model = GNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.MSELoss()

    def train(model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        return loss.item()

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

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping due to no improvement in validation loss")
            break
    """
    #save model

    model_path = os.path.join(model_dir, f"gnn_model_{feature_set_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"GNN model saved to {model_path}")
    """

    model.eval()
    with torch.no_grad():
        y_valid_pred = model(valid_data).squeeze().numpy()
        y_test_pred = model(test_data).squeeze().numpy()

    valid_metrics = evaluate_model(y_valid.numpy(), y_valid_pred, feature_set_name, "GNN")
    test_predictions = {"Predicted": y_test_pred.tolist()}  # Store test predictions

    return model, valid_metrics, test_predictions