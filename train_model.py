import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler
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
    checkpoint = torch.load(filename)
    model = build_transformer(seq_len=input_days, d_model=140, features=num_cols)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch+1}")
    return model, optimizer, epoch

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

# Load dataset
dataset = TensorDataset(features_tensor, labels_tensor)
dataset_avgs = TensorDataset(features_avgs_tensor, labels_avgs_tensor)

# Seed for reproducibility
seed = 42

# Create a RandomSampler with the same seed
sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
sampler_avgs = RandomSampler(dataset_avgs, generator=torch.Generator().manual_seed(seed))

# Create dataloaders with the same sampler
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
dataloader_avgs = DataLoader(dataset_avgs, batch_size=64, sampler=sampler_avgs)

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
checkpoint_file = "checkpoint.pth"

try:
    model, optimizer, start_epoch = load_checkpoint(checkpoint_file)
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")

torch.autograd.set_detect_anomaly(True)

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    progress = 0

    for (batch_features, batch_labels), (batch_features_avgs, batch_labels_avgs) in zip(dataloader, dataloader_avgs):
        optimizer.zero_grad()
        current_features = batch_features.clone()

        # Check for NaN values in data
        if torch.isnan(batch_features).any() or torch.isnan(batch_labels).any() or torch.isnan(batch_features_avgs).any() or torch.isnan(batch_labels_avgs).any():
            continue

        # Normalize input features
        batch_features_normalized = torch.zeros_like(batch_features).to(device)
        batch_labels_normalized = torch.zeros_like(batch_labels).to(device)

        # Loop over each feature
        for i in range(7):
            features_mean = batch_features_avgs[:, i, 0].unsqueeze(1)  # Shape: [32, 1]
            features_std = batch_features_avgs[:, i, 1].unsqueeze(1)   # Shape: [32, 1]
            
            labels_mean = batch_labels_avgs[:, i, 0].unsqueeze(1)      # Shape: [32, 1]
            labels_std = batch_labels_avgs[:, i, 1].unsqueeze(1)       # Shape: [32, 1]

            # Normalize the i-th feature across all samples and time steps
            batch_features_normalized[:, :, i] = (batch_features[:, :, i] - features_mean) / (features_std + 1e-8)
            batch_labels_normalized[:, :, i] = (batch_labels[:, :, i] - labels_mean) / (labels_std + 1e-8)

        # Forward pass
        outputs = model.encode(batch_features_normalized, None)  # Encode the current features
        outputs = model.project(outputs)  # Project the encoded features to the output space

        # Denormalize outputs
        outputs_denormalized = torch.zeros_like(outputs).to(device)

        # Loop over each feature to denormalize
        for i in range(7):
            labels_mean = batch_labels_avgs[:, i, 0].unsqueeze(1).expand_as(outputs[:, :, i])  # Shape: [32, 10]
            labels_std = batch_labels_avgs[:, i, 1].unsqueeze(1).expand_as(outputs[:, :, i])   # Shape: [32, 10]

            # Denormalize the i-th feature across all samples and time steps
            outputs_denormalized[:, :, i] = outputs[:, :, i] * (labels_std + 1e-8) + labels_mean

        # Check for NaN values in outputs
        if torch.isnan(outputs).any() or torch.isnan(outputs_denormalized).any():
            continue

        # Compute the loss comparing the denormalized outputs to the original labels
        loss = criterion(outputs_denormalized, batch_labels)

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

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, checkpoint_file)

torch.save(model.state_dict(), "Minute_Stock_Transformer.pth")
