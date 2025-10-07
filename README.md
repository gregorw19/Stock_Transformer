# Stock Transformer: Transformer-Based Minute-Level Stock Forecasting

This repository implements Transformer-based neural networks for **minute-level stock price forecasting**.  
It includes two main model variants:

1. **Full Sequence Model (`model.py`)** – predicts the next *10 days* (a 10×7 sequence).  
2. **One-Out Model (`model_one.py`)** – predicts only the *next single day* (a 1×7 output) based on a 10-day input.

Both architectures use **Time2Vec positional encoding** and **multi-head self-attention** to capture temporal dependencies and inter-feature relationships in multivariate stock time series.

---

## Repository Structure

```
├── config.py                      # Model hyperparameters and paths
├── dataset_editor.py              # Data preprocessing and normalization script
├── model.py                       # Transformer predicting 10 days of output
├── model_one.py                   # Transformer predicting 1 day of output
├── train_model.py                 # Training script for the 10-day model
├── train_one_out.py               # Training script for the one-out model
├── model_use.py                   # Example inference script
├── Minute_Stock_Transformer.pth   # Weights for 10-day model
├── Minute_Stock_Transformer_One.pth  # Weights for one-out model
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

- **Full Sequence Model:** outputs `(batch_size, 10, 7)` — predicting all 10 days at once  
- **One-Out Model:** outputs `(batch_size, 1, 7)` — predicting only the next day

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

4. **Generate Sliding Windows**  
   Creates overlapping 10-day windows:
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

2. **Model Initialization**  
   Builds the Transformer using:
   ```python
   model = build_transformer(seq_len=10, d_model=140, features=7)
   ```

3. **Loss and Optimizer**
   - Loss: Mean Squared Error (MSE)
   - Optimizer: Adam (learning rate = 1e-4)

4. **Gradient Clipping**
   Stabilizes training with `torch.nn.utils.clip_grad_norm_`.

5. **Checkpointing**
   Saves model and optimizer states every epoch to `checkpoint.pth`.

6. **Model Output**
   - 10-day model → `Minute_Stock_Transformer.pth`
   - One-out model → `Minute_Stock_Transformer_One.pth`

### Example Commands

Train the 10-day model:
```bash
python train_model.py
```

Train the one-out model:
```bash
python train_one_out.py
```

---

## Inference

To run predictions on new data:

```python
from model import build_transformer
import torch
import pandas as pd

# Load model
model = build_transformer(seq_len=10, d_model=140, features=7)
model.load_state_dict(torch.load("Minute_Stock_Transformer.pth", map_location='cpu'))
model.eval()

# Example 10×7 input matrix
data = [
    [89.45, 89.46, 89.37, 89.37, 7872, 2102, 89.424],
    [89.38, 89.53, 89.37, 89.50, 5336, 1938, 89.468],
    ...
]
features_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

# Predict
output = model(features_tensor)
```

To restore original scale:
```python
df_out = (df_pred * stds) + means
```

---

## Output Details

### Full Sequence Model (`model.py`)

- **Input:** `(batch_size, 10, 7)`  
- **Output:** `(batch_size, 10, 7)`  
- **Use case:** multi-step forecasting of the next 10 days

### One-Out Model (`model_one.py`)

- **Input:** `(batch_size, 10, 7)`  
- **Output:** `(batch_size, 1, 7)`  
- **Use case:** single-step forecasting of the next day

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
