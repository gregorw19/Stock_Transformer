import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler
import pandas as pd
from model import *
import itertools

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

# Load and reshape data
data_array = pd.read_csv("train.csv").values.reshape(-1, input_days, num_cols)
labels_array = pd.read_csv("answers.csv").values.reshape(-1, input_days, num_cols)
data_avgs = pd.read_csv("train_avgs.csv").values.reshape(-1, input_days, num_cols)
labels_avgs = pd.read_csv("answers_avgs.csv").values.reshape(-1, input_days, num_cols)

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
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
dataloader_avgs = DataLoader(dataset_avgs, batch_size=32, sampler=sampler_avgs)

print("Datasets loaded")

# Compile model
model = build_transformer(seq_len=input_days, d_model=140, features=num_cols).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("Transformer built")

num_epochs = 50
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

        # Normalize input features
        batch_features_normalized = (batch_features - batch_features_avgs) / (batch_features_avgs + 1e-8)

        # Forward pass
        outputs = model.encode(batch_features_normalized, None)  # Encode the current features
        outputs = model.project(outputs)  # Project the encoded features to the output space

        # Denormalize outputs
        outputs_denormalized = outputs * (batch_labels_avgs + 1e-8) + batch_labels_avgs

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
