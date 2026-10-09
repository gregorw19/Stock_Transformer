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
├── dataset_editor.py              # Builds the training windows from the raw minute data
├── model.py                       # Transformer predicting the next 10 minutes
├── model_one.py                   # Transformer predicting the next minute
├── train_model.py                 # Training script for the 10-minute model
├── train_one_out.py               # Training script for the one-out model
├── evaluate.py                    # Test-set metrics for both models
├── tmodel.py                      # Reference encoder-decoder Transformer for token sequences (not used)
├── test_time2vec.py               # Standalone Time2Vec experiment
├── torch_cuda_check.py            # Prints CUDA availability and GPU info
└── trained_models/
    ├── Minute_Stock_Transformer.pth      # Weights for the 10-minute model
    └── Minute_Stock_Transformer_One.pth  # Weights for the one-out model
```

The raw data lives outside the repository, in a `Data` folder next to it:

```
GitHub/
├── Data/oneMinData/1_min_SPY_2008-2021.csv   # Raw SPY one-minute bars
└── Stock_Transformer/                        # This repository
```

---

## Model Architecture

### Time2Vec Encoding

Both models use **Time2Vec**, a learnable embedding with linear and periodic components.  
For an input \( t \):

$$
\text{Time2Vec}(t) = [w_0 t + b_0, \sin(w_1 t + b_1), \ldots, \sin(w_k t + b_k)]
$$

It is applied to each minute's 8 input columns: the 7 normalized features plus the **time of day** in hours. The time-of-day column lets the model learn daily patterns, such as volume being highest near the open and close.

A **learned position embedding** is added to each minute's encoding. Without it, self-attention treats the 10 minutes as an unordered set and can't tell which one is newest.

### Transformer Encoder

Each model includes multiple encoder layers that consist of:

- **Multi-Head Self-Attention** – captures temporal and inter-feature dependencies  
- **Feed-Forward Network** – applies nonlinear transformations to enhance expressivity  
- **Residual Connections** and **Layer Normalization** – stabilize and accelerate training  

### Output Projection

Both models predict from the encoding of the newest input minute.

- **Full Sequence Model:** outputs `(batch_size, 10, 7)` — predicting all 10 minutes at once  
- **One-Out Model:** outputs `(batch_size, 1, 7)` — predicting only the next minute

Each output is the change in that feature from the last input minute, in units of the input window's standard deviation. An output of 0 means "same as the last minute", which is the baseline the model has to beat.

---

## Data Preprocessing

All preprocessing is handled by `dataset_editor.py`. Run it once before training (it takes under a minute):

```bash
python dataset_editor.py
```

### Steps

1. **Load Data**  
   Reads raw one-minute SPY stock data from:
   ```
   ../Data/oneMinData/1_min_SPY_2008-2021.csv
   ```

2. **Clean**  
   - The raw file stores its trading days in shuffled order and repeats about 640k rows. Rows are de-duplicated and sorted by time, so the training scripts' chronological split really is chronological.
   - Only the regular session is kept: 07:30–13:59 in the file's time zone (9:30–15:59 ET). Some 2020–2021 days also have pre-market and after-hours bars, many with zero volume. Set `regular_hours_only = False` to keep them.

3. **Select Columns**
   ```
   ['open', 'high', 'low', 'close', 'volume', 'barCount', 'average']
   ```

4. **Find Sliding Windows**  
   A window is 10 input minutes followed by the 10 label minutes the model predicts, with one window starting each minute. A window must be 20 consecutive minutes from the same trading day, so no window spans an overnight gap.

5. **Save Output**  
   `../Data/spy_1min_dataset.npz` holds the cleaned, unnormalized bars and the start index of each window (about 57 MB). The training scripts build the windows from it with `load_windows()` in `dataset_editor.py`, which returns:
   ```
   inputs:       (num_windows, 10, 7)
   labels:       (num_windows, 10, 7)
   start_times:  (num_windows,)      first minute of each window
   input_hours:  (num_windows, 10)   time of day of each input minute, in hours (07:30 is 7.5)
   ```
   Volume and barCount come back as log(1 + value). They have rare, huge spikes, and a log scale keeps those from dominating training.

Normalization is done in the training scripts, not here (see below).

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
   Loads the raw input and label windows built by `dataset_editor.py`, in chronological order. The one-out model keeps only the first label minute.

2. **Chronological Train/Validation/Test Split (70/10/20)**  
   The data is a time series of overlapping sliding windows, so it is not split randomly. A random split would put near-copies of each test sample in the training set and make the test loss look better than it really is. Instead:
   - The first 70% of windows are the training set, the next 10% are the validation set, and the last 20% are the test set. The model is always validated and tested on periods after the one it trained on.
   - The validation set picks the best epoch. The test set isn't used during training at all; `evaluate.py` scores it afterwards, so its results are honest.
   - A gap of `2 * input_days` (20) windows is skipped between sets. Each window covers 10 input steps plus 10 label steps, so without the gap the first windows of one set would share minutes with the last windows of the previous set.
   - Batches are shuffled only within the training set. The validation set stays in time order.
   - The date range of each set is printed at startup.

3. **Normalization**    
   Both models normalize each input window by its own per-feature mean and standard deviation, computed in `compute_loss`, then add the time of day as an extra, unnormalized input column. The model predicts each feature's change from the last input minute in units of the window's standard deviation, so its outputs are converted back with `output * std + last input minute`. The label window's statistics are never used, because they describe the future and are not available at prediction time.

4. **Model Initialization**  
   Builds the Transformer using:
   ```python
   model = build_transformer(seq_len=10, d_model=140, features=7)
   ```

5. **Loss and Optimizer**
   - Loss: Mean Squared Error (MSE) in normalized units. Each error is divided by its input window's standard deviation, so every feature counts about equally. In original units, volume (thousands of shares) would swamp prices (dollars), and the model would learn to predict volume and ignore price.
   - That standard deviation is floored at the 5th percentile of each feature's window standard deviation over the training set. Without the floor, a window where the price barely moved would turn even a small move after it into a huge error that swamps the loss.
   - The printed train and validation losses are in these normalized units.
   - Optimizer: Adam (learning rate = 1e-4)

6. **Gradient Clipping**
   Stabilizes training with `torch.nn.utils.clip_grad_norm_`.

7. **Validation**  
   After each epoch the model is evaluated on the validation set (with `model.eval()` and no gradients), and both the train and validation loss are printed:
   ```
   Epoch [1/25], Train Loss: ..., Val Loss: ...
   ```

8. **Checkpointing**
   Saves model and optimizer states, and the best validation loss so far, every epoch. Each script uses its own checkpoint file so it never resumes from the other script's model, or from an old `checkpoint.pth` trained before the train/test split:
   - 10-minute model → `checkpoint_train_model.pth`
   - One-out model → `checkpoint_train_one_out.pth`

   When a checkpoint is found, training resumes at the epoch after the one that was saved.

   Delete a script's checkpoint file to start a fresh training run, for example after rebuilding the dataset or changing the model.

9. **Model Output**  
   Whenever the validation loss reaches a new low, the weights are saved to:
   - 10-minute model → `trained_models/Minute_Stock_Transformer.pth`
   - One-out model → `trained_models/Minute_Stock_Transformer_One.pth`

   So the file always holds the best epoch, not the last one. Training overwrites the file of the same name, so copy the old weights elsewhere first if you want to keep them.

All paths are relative to the scripts, so they can be run from any working directory.

> **Note:** Weights trained before the position embedding, time-of-day input and newest-minute projection were added have a different architecture and won't load. Retrain both models.

### Example Commands

Build the dataset (once):
```bash
python dataset_editor.py
```

Train the 10-minute model (about 10 minutes per epoch on an RTX 3090):
```bash
python train_model.py
```

Train the one-out model (about 20 minutes per epoch, since it uses batches of 32 instead of 64):
```bash
python train_one_out.py
```

---

## Evaluation

`evaluate.py` scores the trained models on the same chronological test set the training scripts hold out (the last 20% of windows). Each model is run exactly the way its training script runs it.

```bash
python evaluate.py
```

By default it evaluates both models in `trained_models/` and skips any that haven't been trained yet. To evaluate one model, or a specific weights file, pass `--model` and `--weights`. `--weights` also accepts a training checkpoint, so you can check a model partway through training:

```bash
python evaluate.py --model one_out --weights checkpoint_train_one_out.pth
```

### What it reports

Every model is compared with a simple **baseline**:
- Every future price (open, high, low, close, average) equals the last input close. A minute's open is almost always the previous minute's close, so this is fairer than repeating each price's own last value.
- Volume and barCount equal their average over the 10 input minutes.

Minute-level prices barely move, so this simple guess is hard to beat. It is the bar a model has to clear to be useful. Volume and barCount are reported in original units (shares and trades), not on the log scale used for training.

- **Per feature** (over all predicted minutes): MAE and RMSE in original units for the model and the baseline, plus **R2 vs base** = 1 − model MSE / baseline MSE. Above 0 means the model beats the baseline; below 0 means it does worse. Plain R² isn't used because prices are so autocorrelated that any reasonable guess scores about 0.9999.
- **Close price by minute ahead**: MAE for the model and baseline, R2 vs base, and **direction accuracy**: how often the model correctly predicts whether the close will be above or below the last input close. Windows where the close didn't move are skipped. **Always up** is the accuracy of always predicting a rise, so a model needs to beat both it and 50% to have learned anything about direction.

---

## Inference

The `Transformer` class has no `forward()` method, so call `encode` and then `project`, the same way the training scripts do.

### 10-minute model

Prepare the input exactly the way the training scripts do: log scale for volume and barCount, normalization by each window's own per-feature mean and standard deviation, and the time of day as an 8th column. The output is each feature's change from the last input minute.

```python
import torch
from model import build_transformer

model = build_transformer(seq_len=10, d_model=140, features=7)
model.load_state_dict(torch.load("trained_models/Minute_Stock_Transformer.pth", map_location="cpu"))
model.eval()

# The last 10 one-minute bars, oldest first:
# open, high, low, close, volume, barCount, average
data = [
    [89.45, 89.46, 89.37, 89.37, 7872, 2102, 89.424],
    [89.38, 89.53, 89.37, 89.50, 5336, 1938, 89.468],
    ...  # 8 more rows
]
x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, 10, 7)
x[..., 4:6] = torch.log1p(x[..., 4:6])  # volume and barCount on a log scale

# Time of day of each bar in hours, in the data file's time zone (07:30 is 7.5)
hours = torch.tensor([7.5 + i / 60 for i in range(10)]).view(1, 10, 1)

# Normalize each feature by this window's own mean and standard deviation,
# then add the time of day as an 8th column
means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
x_norm = torch.cat([(x - means) / (stds + 1e-8), hours], dim=-1)  # (1, 10, 8)

with torch.no_grad():
    out = model.project(model.encode(x_norm, None))  # (1, 10, 7)

# The output is the change from the last input minute, in units of the window's std
prediction = out * (stds + 1e-8) + x[:, -1:, :]
prediction[..., 4:6] = torch.expm1(prediction[..., 4:6])  # back to shares and trades
```

This is the same normalization `compute_loss` in the training scripts uses during training (including `torch.std`, which is the sample standard deviation).

### One-out model

This model uses the same input preparation, so reuse `x`, `x_norm` and `stds` from above:

```python
from model_one import build_transformer

model = build_transformer(seq_len=10, d_model=140, features=7)
model.load_state_dict(torch.load("trained_models/Minute_Stock_Transformer_One.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    out = model.project(model.encode(x_norm, None))  # (1, 1, 7)

prediction = out * (stds + 1e-8) + x[:, -1:, :]
prediction[..., 4:6] = torch.expm1(prediction[..., 4:6])
```

---

## Output Details

### Full Sequence Model (`model.py`)

- **Input:** `(batch_size, 10, 8)` (7 normalized features + time of day)  
- **Output:** `(batch_size, 10, 7)`  
- **Use case:** multi-step forecasting of the next 10 minutes

### One-Out Model (`model_one.py`)

- **Input:** `(batch_size, 10, 8)` (7 normalized features + time of day)  
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
- Extend dataset to multiple tickers and intervals

---

## License

This project is for **research and educational purposes only**.  
It is not intended for live trading or financial decision-making.
