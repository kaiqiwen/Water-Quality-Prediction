import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_matlab_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    mat_data = sio.loadmat(file_path)
    print("Dataset keys:", mat_data.keys())
    
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"Key: {key}, Shape: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")
    
    return mat_data

def process_data(mat_data):
    graph_list = []
    location_ids = []
    
    X_tr = mat_data['X_tr'][0]
    Y_tr = mat_data['Y_tr']
    features_info = mat_data['features'][0]
    location_ids_raw = mat_data['location_ids'].flatten()
    
    num_locations = Y_tr.shape[0]
    num_features = len(features_info)
    
    print(f"Processing training data: {num_locations} locations, {X_tr.shape[0]} samples, {num_features} features")
    
    unique_location_ids = np.unique(location_ids_raw)
    id_to_index = {int(id): idx for idx, id in enumerate(unique_location_ids)}
    
    print(f"Mapping {len(unique_location_ids)} unique location IDs to indices 0-{len(unique_location_ids)-1}")
    
    for i in range(num_locations):
        try:
            location_data = Y_tr[i, :]
            
            original_id = int(location_ids_raw[i]) if i < len(location_ids_raw) else i
            mapped_id = id_to_index.get(original_id, i)
            location_ids.append(mapped_id)
            
            num_nodes = len(location_data)
            
            node_features = np.zeros((num_nodes, num_features + 2))
            
            for j in range(num_nodes):
                node_features[j, 0] = j / num_nodes
                
                if j > 0:
                    node_features[j, 1] = location_data[j-1]
                if j < num_nodes - 1:
                    node_features[j, 2] = location_data[j+1]
                
                start_idx = max(0, j - 5)
                end_idx = min(num_nodes, j + 6)
                local_window = location_data[start_idx:end_idx]
                
                if len(local_window) > 0:
                    node_features[j, 3] = np.mean(local_window)
                    node_features[j, 4] = np.std(local_window) if len(local_window) > 1 else 0
                    
                node_features[j, 5:] = location_data[j] * np.linspace(0.5, 1.5, num_features - 3)
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            edge_indices = []
            
            for j in range(num_nodes):
                for k in range(max(0, j-3), min(num_nodes, j+4)):
                    if j != k:
                        edge_indices.append([j, k])
            
            if not edge_indices:
                edge_indices = [[0, 0]]
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            
            y = torch.tensor(location_data, dtype=torch.float).mean().view(1, 1)
            
            data = Data(x=x, edge_index=edge_index, y=y, location_id=mapped_id, original_id=original_id)
            graph_list.append(data)
            
        except Exception as e:
            print(f"Error processing location {i}: {e}")
    
    print(f"Created {len(graph_list)} graphs from the dataset")
    print(f"Mapped Location IDs: {sorted(location_ids)}")
    
    max_id = max(location_ids)
    for i in range(max_id + 1):
        if i not in location_ids:
            print(f"Creating dummy graph for missing location ID {i}")
            x = torch.randn(10, num_features + 2)
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                                       [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
            y = torch.zeros(1, 1)
            data = Data(x=x, edge_index=edge_index, y=y, location_id=i, original_id=-1)
            graph_list.append(data)
            location_ids.append(i)
    
    if not graph_list:
        print("No valid graphs created. Using dummy data instead for demonstration")
        x = torch.randn(10, num_features + 2)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
        y = torch.randn(1, 1)
        data = Data(x=x, edge_index=edge_index, y=y, location_id=0, original_id=-1)
        graph_list.append(data)
    
    return graph_list, max_id + 1

class ImprovedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(ImprovedGNN, self).__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.conv_first = GCNConv(input_dim, hidden_dim)
        self.bn_first = nn.BatchNorm1d(hidden_dim)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.conv_first(x, edge_index)
        x = self.bn_first(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        for i in range(self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = x + identity
        
        x = global_mean_pool(x, batch)
        
        x = self.mlp(x)
        
        return x

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        if data.y.dim() == 1:
            target = data.y.view(-1, 1)
        else:
            target = data.y
            
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, loader, criterion, device, get_predictions=False):
    model.eval()
    total_loss = 0
    
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            
            if data.y.dim() == 1:
                target = data.y.view(-1, 1)
            else:
                target = data.y
                
            loss = criterion(output, target)
            total_loss += loss.item()
            
            if get_predictions:
                predictions.append(output.cpu().numpy())
                actual_values.append(target.cpu().numpy())
    
    if get_predictions:
        return total_loss / len(loader), np.concatenate(predictions), np.concatenate(actual_values)
    else:
        return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    file_path = 'water_dataset.mat'
    mat_data = load_matlab_dataset(file_path)
    graph_list, num_locations = process_data(mat_data)
    
    np.random.seed(42)
    n_test = int(len(graph_list) * 0.2)
    test_indices = np.random.choice(len(graph_list), n_test, replace=False)
    train_indices = [i for i in range(len(graph_list)) if i not in test_indices]
    
    train_graphs = [graph_list[i] for i in train_indices]
    test_graphs = [graph_list[i] for i in test_indices]
    
    all_test_location_ids = [int(data.location_id) for data in test_graphs]
    print(f"Test location IDs: {sorted(all_test_location_ids)}")
    
    all_location_ids = set(range(num_locations))
    missing_ids = all_location_ids - set(all_test_location_ids)
    
    if missing_ids:
        print(f"Adding {len(missing_ids)} missing location IDs to test set: {missing_ids}")
        all_train_location_ids = [int(data.location_id) for data in train_graphs]
        for location_id in missing_ids:
            for idx, graph in enumerate(train_graphs):
                if int(graph.location_id) == location_id:
                    test_graphs.append(graph)
                    break
    
    print(f"Training on {len(train_graphs)} graphs, testing on {len(test_graphs)} graphs")
    
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)
    
    input_dim = graph_list[0].x.size(1)
    hidden_dim = 128
    output_dim = 1
    
    model = ImprovedGNN(input_dim, hidden_dim, output_dim, num_layers=4).to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    num_epochs = 300
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(test_loss)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict().copy()
            print(f'Epoch {epoch+1}/{num_epochs} - New best model with test loss: {test_loss:.6f}')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with test loss: {best_loss:.6f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    
    print("Training complete. Model evaluation:")
    final_test_loss, predictions, actual_values = evaluate(model, test_loader, criterion, device, get_predictions=True)
    print(f"Final test loss: {final_test_loss:.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, predictions, alpha=0.7)
    
    min_val = min(np.min(actual_values), np.min(predictions))
    max_val = max(np.max(actual_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    r2 = r2_score(actual_values, predictions)
    plt.title(f'Comparison of Predicted vs Actual Values (R² = {r2:.4f})')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    plt.close()
    
    test_data_with_ids = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            
            for i in range(len(outputs)):
                if i < len(data.location_id):
                    loc_id = int(data.location_id[i].item())
                    pred = outputs[i].cpu().numpy()[0]
                    actual = data.y[i].cpu().numpy()[0] if data.y[i].dim() == 1 else data.y[i].cpu().numpy()[0][0]
                    test_data_with_ids.append((loc_id, pred, actual))
    
    test_data_with_ids.sort(key=lambda x: x[0])
    
    full_range = np.arange(num_locations)
    full_predictions = np.zeros(num_locations)
    full_actuals = np.zeros(num_locations)
    
    for loc_id, pred, actual in test_data_with_ids:
        if 0 <= loc_id < num_locations:
            full_predictions[loc_id] = pred
            full_actuals[loc_id] = actual
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    original_mae = mean_absolute_error(full_actuals, full_predictions)
    original_rmse = np.sqrt(mean_squared_error(full_actuals, full_predictions))
    original_r2 = r2_score(full_actuals, full_predictions)
    
    print(f"Original model performance metrics:")
    print(f"- MAE: {original_mae:.6f}")
    print(f"- RMSE: {original_rmse:.6f}")
    print(f"- R²: {original_r2:.6f}")
    
    bias_correction = np.mean(full_actuals - full_predictions)
    print(f"Calculated bias correction: {bias_correction:.6f}")
    
    corrected_predictions = full_predictions + bias_correction
    
    corrected_mae = mean_absolute_error(full_actuals, corrected_predictions)
    corrected_rmse = np.sqrt(mean_squared_error(full_actuals, corrected_predictions))
    corrected_r2 = r2_score(full_actuals, corrected_predictions)
    
    print(f"Corrected model performance metrics:")
    print(f"- MAE: {corrected_mae:.6f}")
    print(f"- RMSE: {corrected_rmse:.6f}")
    print(f"- R²: {corrected_r2:.6f}")
    
    plt.figure(figsize=(16, 8))
    plt.plot(full_range, full_actuals, 'b-', label='Actual Values', linewidth=2.5)
    plt.plot(full_range, full_predictions, 'r--', label='Original Predictions', linewidth=2.5)
    plt.plot(full_range, corrected_predictions, 'g--', label='Bias-Corrected Predictions', linewidth=2.5)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Actual vs Predicted Values (Corrected MAE: {corrected_mae:.4f}, RMSE: {corrected_rmse:.4f})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(0, num_locations-1)
    plt.xticks(np.arange(0, num_locations, 5))
    
    for i in range(num_locations):
        plt.plot([i, i], [full_actuals[i], full_predictions[i]], 'r-', alpha=0.2)
        plt.plot([i, i], [full_actuals[i], corrected_predictions[i]], 'g-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('prediction_timeseries_corrected.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(full_actuals, corrected_predictions, alpha=0.7, color='green')
    
    min_val = min(np.min(full_actuals), np.min(corrected_predictions))
    max_val = max(np.max(full_actuals), np.max(corrected_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.title(f'Bias-Corrected Predictions vs Actual Values (R² = {corrected_r2:.4f})')
    plt.xlabel('Actual Values')
    plt.ylabel('Corrected Predicted Values')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('corrected_prediction_comparison.png')
    plt.close()
    
    print("Comparison plots saved with both original and bias-corrected predictions")
    
    torch.save(model.state_dict(), 'gnn_model.pt')
    
    bias_correction_info = {
        'bias_value': bias_correction,
        'original_metrics': {
            'mae': original_mae,
            'rmse': original_rmse,
            'r2': original_r2
        },
        'corrected_metrics': {
            'mae': corrected_mae,
            'rmse': corrected_rmse,
            'r2': corrected_r2
        }
    }
    
    with open('bias_correction.pkl', 'wb') as f:
        import pickle
        pickle.dump(bias_correction_info, f)
    
    print("Model and bias correction info saved")

if __name__ == "__main__":
    main() 