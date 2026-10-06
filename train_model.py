import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
import pandas as pd
from model import *
import itertools
import numpy as np

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(filename="checkpoint.pth"):
    checkpoint = torch.load(filename, map_location=device)
    model = build_transformer(seq_len=input_days, d_model=140, features=num_cols).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch+1}")
    # The saved epoch already finished, so resume from the next one
    return model, optimizer, epoch + 1

class StockDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Configuration
input_days = 10
num_cols = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the list of columns you want to select
columns = ["Mean1", "Standard_Deviation1", "Mean2", "Standard_Deviation2", 
           "Mean3", "Standard_Deviation3", "Mean4", "Standard_Deviation4", 
           "Mean5", "Standard_Deviation5", "Mean6", "Standard_Deviation6", 
           "Mean7", "Standard_Deviation7"]

# Load data
data_array = pd.read_csv("Stock_Transformer/train.csv").values
labels_array = pd.read_csv("Stock_Transformer/answers.csv").values
data_avgs = pd.read_csv("Stock_Transformer/train_avgs.csv")[columns].values
labels_avgs = pd.read_csv("Stock_Transformer/answers_avgs.csv")[columns].values

# Reshape arrays
data_array = data_array.reshape(2070805, input_days, num_cols)  # (2070805, 10, 7)
labels_array = labels_array.reshape(2070805, input_days, num_cols)
data_avgs = data_avgs.reshape(2070805, num_cols, 2)  # (2070805, 7, 2)
labels_avgs = labels_avgs.reshape(2070805, num_cols, 2)
print("Arrays Loaded")

print(f"data_array shape after reshape: {data_array.shape}")
print(f"labels_array shape after reshape: {labels_array.shape}")
print(f"data_avgs shape after reshape: {data_avgs.shape}")
print(f"labels_avgs shape after reshape: {labels_avgs.shape}")

# Convert dataframes to tensors and move to GPU if available
features_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels_array, dtype=torch.float32).to(device)
features_avgs_tensor = torch.tensor(data_avgs, dtype=torch.float32).to(device)
labels_avgs_tensor = torch.tensor(labels_avgs, dtype=torch.float32).to(device)

# Load dataset (features, labels and their avgs stay aligned in one dataset)
dataset = TensorDataset(features_tensor, labels_tensor, features_avgs_tensor, labels_avgs_tensor)

# Chronological 80/20 split: train on the earlier 80%, test on the later 20%.
# Windows overlap with stride 1, so skip a gap of 2 * input_days samples so no
# test window shares any minutes with a training window's inputs or labels.
num_samples = len(dataset)
split_idx = int(num_samples * 0.8)
gap = 2 * input_days
train_dataset = Subset(dataset, range(0, split_idx))
test_dataset = Subset(dataset, range(split_idx + gap, num_samples))
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# Seed for reproducibility
seed = 42

# Shuffle only within the training set; keep the test set in time order
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Datasets loaded")

# Compile model
model = build_transformer(seq_len=input_days, d_model=140, features=num_cols).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("Transformer built")

num_epochs = 25
clip_value = 1.0  # Gradient clipping value

# To load from a checkpoint
start_epoch = 0
# Per-script name, so old checkpoints (trained on all data, before the
# train/test split) and the other training script's checkpoints are never resumed
checkpoint_file = "checkpoint_train_model.pth"

try:
    model, optimizer, start_epoch = load_checkpoint(checkpoint_file)
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")

def compute_loss(batch_features, batch_labels, batch_features_avgs, batch_labels_avgs):
    """Runs a forward pass on one batch. Returns None if the batch contains NaNs."""
    # Check for NaN values in data
    if torch.isnan(batch_features).any() or torch.isnan(batch_labels).any() or torch.isnan(batch_features_avgs).any() or torch.isnan(batch_labels_avgs).any():
        return None

    # Normalize input features
    batch_features_normalized = torch.zeros_like(batch_features).to(device)

    # Loop over each feature
    for i in range(7):
        features_mean = batch_features_avgs[:, i, 0].unsqueeze(1)  # Shape: [32, 1]
        features_std = batch_features_avgs[:, i, 1].unsqueeze(1)   # Shape: [32, 1]

        # Normalize the i-th feature across all samples and time steps
        batch_features_normalized[:, :, i] = (batch_features[:, :, i] - features_mean) / (features_std + 1e-8)

    # Forward pass
    outputs = model.encode(batch_features_normalized, None)  # Encode the current features
    outputs = model.project(outputs)  # Project the encoded features to the output space

    # Denormalize outputs
    outputs_denormalized = torch.zeros_like(outputs).to(device)

    # Loop over each feature to denormalize. Use the input window's stats: the label
    # window's stats describe the future and aren't available at prediction time.
    for i in range(7):
        features_mean = batch_features_avgs[:, i, 0].unsqueeze(1).expand_as(outputs[:, :, i])  # Shape: [32, 10]
        features_std = batch_features_avgs[:, i, 1].unsqueeze(1).expand_as(outputs[:, :, i])   # Shape: [32, 10]

        # Denormalize the i-th feature across all samples and time steps
        outputs_denormalized[:, :, i] = outputs[:, :, i] * (features_std + 1e-8) + features_mean

    # Check for NaN values in outputs
    if torch.isnan(outputs).any() or torch.isnan(outputs_denormalized).any():
        return None

    # Compute the loss comparing the denormalized outputs to the original labels
    return criterion(outputs_denormalized, batch_labels)

def evaluate(loader):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            loss = compute_loss(*batch)
            if loss is None:
                continue
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)

torch.autograd.set_detect_anomaly(True)

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    progress = 0

    for batch in train_loader:
        optimizer.zero_grad()

        loss = compute_loss(*batch)
        if loss is None:
            continue

        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Optimization step
        optimizer.step()

        progress += 1
        if progress % 1000 == 0:
            print(f'Progress: {progress} batches processed')

    avg_loss = total_loss / len(train_loader)
    test_loss = evaluate(test_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, checkpoint_file)

torch.save(model.state_dict(), "Minute_Stock_Transformer.pth")
