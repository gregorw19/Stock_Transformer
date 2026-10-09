import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from pathlib import Path
import pandas as pd
from model import *
from dataset_editor import load_windows, input_days
import itertools
import numpy as np

def save_checkpoint(model, optimizer, epoch, best_val_loss, filename="checkpoint.pth"):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
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
    return model, optimizer, epoch + 1, checkpoint['best_val_loss']

class StockDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Configuration
num_cols = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths are relative to this file, so the script works from any working directory
project_dir = Path(__file__).resolve().parent

# Load the raw input and label windows built by dataset_editor.py
data_array, labels_array, window_times, hours_array = load_windows()
print("Arrays Loaded")

print(f"data_array shape: {data_array.shape}")  # (num_windows, 10, 7)
print(f"labels_array shape: {labels_array.shape}")  # (num_windows, 10, 7)

# Convert arrays to tensors and move to GPU if available
features_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels_array, dtype=torch.float32).to(device)
hours_tensor = torch.tensor(hours_array, dtype=torch.float32).to(device)

# Load dataset (features, labels and times of day stay aligned in one dataset)
dataset = TensorDataset(features_tensor, labels_tensor, hours_tensor)

# Chronological 70/10/20 split: train on the earliest 70%, pick the best epoch on the
# next 10% (validation), and leave the last 20% untouched for evaluate.py (test).
# Windows overlap with stride 1, so skip a gap of 2 * input_days samples between sets
# so no window shares any minutes with a window in another set.
num_samples = len(dataset)
val_idx = int(num_samples * 0.7)
split_idx = int(num_samples * 0.8)
gap = 2 * input_days
train_dataset = Subset(dataset, range(0, val_idx))
val_dataset = Subset(dataset, range(val_idx + gap, split_idx))
test_dataset = Subset(dataset, range(split_idx + gap, num_samples))
print(f"Train samples: {len(train_dataset)} ({window_times[0]} to {window_times[val_idx - 1]})")
print(f"Validation samples: {len(val_dataset)} ({window_times[val_idx + gap]} to {window_times[split_idx - 1]})")
print(f"Test samples: {len(test_dataset)} ({window_times[split_idx + gap]} to {window_times[-1]})")

# Floor for the loss scale in compute_loss: the 5th percentile of each feature's
# input-window std over the training set. Without it, a near-flat window (std close
# to 0) turns even a small move after it into a huge error that swamps the loss.
train_stds = data_array[:val_idx].std(axis=1, ddof=1)  # Sample std, same as torch.std
std_floor = torch.tensor(np.percentile(train_stds, 5, axis=0), dtype=torch.float32, device=device)  # Shape: [7]

# Seed for reproducibility
seed = 42

# Shuffle only within the training set; keep the validation set in time order
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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
checkpoint_file = project_dir / "checkpoint_train_model.pth"
# The weights from the epoch with the lowest validation loss are saved here
model_file = project_dir / "trained_models" / "Minute_Stock_Transformer.pth"
model_file.parent.mkdir(exist_ok=True)
best_val_loss = float('inf')

try:
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(checkpoint_file)
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")

def compute_loss(batch_features, batch_labels, batch_hours):
    """Runs a forward pass on one batch. Returns None if the batch contains NaNs."""
    # Check for NaN values in data
    if torch.isnan(batch_features).any() or torch.isnan(batch_labels).any():
        return None

    # Normalize each input window by its own per-feature mean and standard deviation
    features_mean = batch_features.mean(dim=1, keepdim=True)  # Shape: [batch, 1, 7]
    features_std = batch_features.std(dim=1, keepdim=True)    # Shape: [batch, 1, 7]
    batch_features_normalized = (batch_features - features_mean) / (features_std + 1e-8)

    # Add each minute's time of day as an extra, unnormalized input column
    model_inputs = torch.cat([batch_features_normalized, batch_hours.unsqueeze(-1)], dim=-1)  # Shape: [batch, 10, 8]

    # Forward pass
    outputs = model.encode(model_inputs, None)  # Encode the current features
    outputs = model.project(outputs)  # Project the encoded features to the output space

    # The model predicts each feature's change from the last input minute, in units of
    # the input window's std, so an output of 0 means "same as the last minute". Use the
    # input window's stats: the label window's stats aren't available at prediction time.
    outputs_denormalized = outputs * (features_std + 1e-8) + batch_features[:, -1:, :]

    # Check for NaN values in outputs
    if torch.isnan(outputs).any() or torch.isnan(outputs_denormalized).any():
        return None

    # Compute the loss in normalized units: divide each error by the input window's std
    # (floored), so every feature counts about equally. In original units, volume
    # (thousands of shares) would swamp prices (dollars).
    loss_scale = torch.maximum(features_std, std_floor)
    return criterion(outputs_denormalized / loss_scale, batch_labels / loss_scale)

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

# Debugging only: makes training ~15x slower. Uncomment to trace NaNs in the backward pass.
# torch.autograd.set_detect_anomaly(True)

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
    val_loss = evaluate(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Keep the weights from the epoch with the lowest validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_file)
        print(f"New best validation loss, saved to {model_file}")

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_file)

print(f"Best validation loss: {best_val_loss:.4f}, weights in {model_file}")
