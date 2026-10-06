# Stock Transformer: Transformer-Based Minute-Level Stock Forecasting

This repository implements Transformer-based neural networks for **minute-level stock price forecasting**.  
It includes two main model variants:

1. **Full Sequence Model (`model.py`)** – predicts the next 10 minutes (a 10×7 sequence) from the previous 10 one-minute bars.  
2. **One-Out Model (`model_one.py`)** – predicts only the next minute (a 1×7 output) from the previous 10 one-minute bars.

Both architectures use **Time2Vec positional encoding** and **multi-head self-attention** to capture temporal dependencies and inter-feature relationships in multivariate stock time series.

---

## Repository Structure

```
├── config.py                      # Model hyperparameters and paths
├── dataset_editor.py              # Data preprocessing and normalization script
├── model.py                       # Transformer predicting the next 10 minutes
├── model_one.py                   # Transformer predicting the next minute
├── train_model.py                 # Training script for the 10-minute model
├── train_one_out.py               # Training script for the one-out model
├── tmodel.py                      # Reference encoder-decoder Transformer for token sequences (not used)
├── test_time2vec.py               # Standalone Time2Vec experiment
├── torch_cuda_check.py            # Prints CUDA availability and GPU info
├── Minute_Stock_Transformer.pth   # Weights for the 10-minute model
├── Minute_Stock_Transformer_One.pth  # Weights for the one-out model
```

---

## Model Architecture

### Time2Vec Encoding

Both models use **Time2Vec**, a learnable temporal embedding that replaces fixed sinusoidal positional encodings.  
For a time input \( t \):

$$
\text{Time2Vec}(t) = [w_0 t + b_0, \sin(w_1 t + b_1), \ldots, \sin(w_k t + b_k)]
$$

This allows the model to capture both **linear** and **periodic** components of time.

### Transformer Encoder

Each model includes multiple encoder layers that consist of:

- **Multi-Head Self-Attention** – captures temporal and inter-feature dependencies  
- **Feed-Forward Network** – applies nonlinear transformations to enhance expressivity  
- **Residual Connections** and **Layer Normalization** – stabilize and accelerate training  

### Output Projection

- **Full Sequence Model:** outputs `(batch_size, 10, 7)` — predicting all 10 minutes at once  
- **One-Out Model:** outputs `(batch_size, 1, 7)` — predicting only the next minute

---

## Data Preprocessing

All preprocessing is handled by `dataset_editor.py`.

### Steps

1. **Load Data**  
   Reads raw one-minute SPY stock data from:
   ```
   Stock_Transformer/Data/oneMinData/1_min_SPY_2008-2021.csv
   ```

2. **Select Columns**
   ```
   ['open', 'high', 'low', 'close', 'volume', 'barCount', 'average']
   ```

3. **Standardize Each Column**  
   Each feature is standardized using z-score normalization:
   $$
   z = \frac{x - \mu}{\sigma}
   $$
   The mean and standard deviation are computed from the **first 80% of rows only** (the training period) and then applied to all rows, so no information from the test period leaks into the scaling.

4. **Generate Sliding Windows**  
   Creates overlapping 10-minute windows (one window per minute):
   ```
   (num_samples, 10, 7)
   ```

5. **Save Outputs**  
   - `standardized_data.csv` — standardized sliding windows  
   - `means_stds.csv` — per-column means and standard deviations for inference

---

## Configuration

The configuration file `config.py` defines experiment parameters and paths:

```python
{
    "batch_size": 8,
    "num_epochs": 25,
    "lr": 1e-4,
    "seq_len": 10,
    "d_model": 512,
    "num_features": 7,
    "model_folder": "weights",
    "model_basename": "s_model",
    "experiment_name": "runs/tmodel"
}
```

---

## Training

### Workflow

1. **Data Loading**  
   Loads preprocessed `train.csv`, `answers.csv`, and their average normalization files.

2. **Chronological Train/Test Split (80/20)**  
   The data is a time series of overlapping sliding windows, so it is **not** split randomly. A random split would put near-copies of each test sample in the training set and make the test loss look better than it really is. Instead:
   - The first 80% of windows are the training set and the last 20% are the test set, so the model is always tested on a period after the one it trained on.
   - A gap of `2 * input_days` (20) windows is skipped between the two sets. Each window covers 10 input steps plus 10 label steps, so without the gap the first test windows would share minutes with the last training windows.
   - Batches are shuffled only within the training set. The test set stays in time order.

3. **Normalization (10-minute model)**    
   Each input window is normalized by its own per-feature mean and standard deviation. The model's outputs are converted back to the original scale with the **same input-window statistics**. The label window's statistics are never used, because they describe the future and are not available at prediction time.

4. **Model Initialization**  
   Builds the Transformer using:
   ```python
   model = build_transformer(seq_len=10, d_model=140, features=7)
   ```

5. **Loss and Optimizer**
   - Loss: Mean Squared Error (MSE)
   - Optimizer: Adam (learning rate = 1e-4)

6. **Gradient Clipping**
   Stabilizes training with `torch.nn.utils.clip_grad_norm_`.

7. **Evaluation**  
   After each epoch the model is evaluated on the test set (with `model.eval()` and no gradients), and both the train and test loss are printed:
   ```
   Epoch [1/25], Train Loss: ..., Test Loss: ...
   ```

8. **Checkpointing**
   Saves model and optimizer states every epoch. Each script uses its own checkpoint file so it never resumes from the other script's model, or from an old `checkpoint.pth` trained before the train/test split:
   - 10-minute model → `checkpoint_train_model.pth`
   - One-out model → `checkpoint_train_one_out.pth`

   When a checkpoint is found, training resumes at the epoch after the one that was saved.

9. **Model Output**
   - 10-minute model → `Minute_Stock_Transformer.pth`
   - One-out model → `Minute_Stock_Transformer_One.pth`

> **Note:** The included `.pth` weights were trained before the train/test split and normalization changes, on the full dataset. Retrain both models before relying on any test results.

### Example Commands

Train the 10-minute model:
```bash
python train_model.py
```

Train the one-out model:
```bash
python train_one_out.py
```

---

## Inference

The `Transformer` class has no `forward()` method, so call `encode` and then `project`, the same way the training scripts do.

### 10-minute model

This model was trained on inputs normalized by each window's own per-feature mean and standard deviation, so do the same at inference and use those statistics to convert the output back to the original scale.

```python
import torch
from model import build_transformer

model = build_transformer(seq_len=10, d_model=140, features=7)
model.load_state_dict(torch.load("Minute_Stock_Transformer.pth", map_location="cpu"))
model.eval()

# The last 10 one-minute bars, oldest first:
# open, high, low, close, volume, barCount, average
data = [
    [89.45, 89.46, 89.37, 89.37, 7872, 2102, 89.424],
    [89.38, 89.53, 89.37, 89.50, 5336, 1938, 89.468],
    ...  # 8 more rows
]
x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, 10, 7)

# Normalize each feature by this window's own mean and standard deviation
means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
x_norm = (x - means) / (stds + 1e-8)

with torch.no_grad():
    out = model.project(model.encode(x_norm, None))  # (1, 10, 7)

# Convert back to the original scale with the same input-window statistics
prediction = out * (stds + 1e-8) + means
```

The mean and standard deviation should be computed the same way as the values in `train_avgs.csv` that the model was trained with.

### One-out model

This model was trained on raw, unnormalized values, so the input goes in as-is and the output is already in the original scale.

```python
from model_one import build_transformer

model = build_transformer(seq_len=10, d_model=140, features=7)
model.load_state_dict(torch.load("Minute_Stock_Transformer_One.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    prediction = model.project(model.encode(x, None))  # (1, 1, 7)
```

---

## Output Details

### Full Sequence Model (`model.py`)

- **Input:** `(batch_size, 10, 7)`  
- **Output:** `(batch_size, 10, 7)`  
- **Use case:** multi-step forecasting of the next 10 minutes

### One-Out Model (`model_one.py`)

- **Input:** `(batch_size, 10, 7)`  
- **Output:** `(batch_size, 1, 7)`  
- **Use case:** single-step forecasting of the next minute

---

## Requirements

Install dependencies:

```bash
pip install torch pandas numpy
```

---

## Example Columns

The model expects data in this order:

```
Open, High, Low, Close, Volume, Trades, VWAP
```

---

## Future Work

- Add decoder for autoregressive forecasting  
- Visualize attention weights  
- Evaluate with MAE, RMSE, and R² metrics  
- Extend dataset to multiple tickers and intervals

---

## License

This project is for **research and educational purposes only**.  
It is not intended for live trading or financial decision-making.
