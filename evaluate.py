import argparse
from pathlib import Path
import numpy as np
import torch
from dataset_editor import load_windows, input_days, columns, log_columns
from model import build_transformer as build_ten_minute_model
from model_one import build_transformer as build_one_out_model

# Paths are relative to this file, so the script works from any working directory
project_dir = Path(__file__).resolve().parent
default_weights = {
    "ten_minute": project_dir / "trained_models" / "Minute_Stock_Transformer.pth",
    "one_out": project_dir / "trained_models" / "Minute_Stock_Transformer_One.pth",
}

num_cols = 7
close_col = columns.index('close')
batch_size = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name, weights_path):
    build = build_ten_minute_model if name == "ten_minute" else build_one_out_model
    model = build(seq_len=input_days, d_model=140, features=num_cols).to(device)
    state = torch.load(weights_path, map_location=device)
    # Accept a final .pth (a state dict) or a training checkpoint (a dict holding one)
    model.load_state_dict(state.get('model_state_dict', state))
    model.eval()
    return model


def predict(model, inputs, input_hours):
    """Runs the model the same way the training scripts do.

    Returns predictions in the same scale as inputs (volume and barCount still on a
    log scale), shape (num_windows, 10, 7) for the 10-minute model or
    (num_windows, 1, 7) for the one-out model.
    """
    preds = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.tensor(inputs[start:start + batch_size], dtype=torch.float32, device=device)
            hours = torch.tensor(input_hours[start:start + batch_size], dtype=torch.float32, device=device)
            # Same per-window normalization and time-of-day column as compute_loss in the training scripts
            features_mean = x.mean(dim=1, keepdim=True)
            features_std = x.std(dim=1, keepdim=True)
            model_inputs = torch.cat([(x - features_mean) / (features_std + 1e-8), hours.unsqueeze(-1)], dim=-1)
            outputs = model.project(model.encode(model_inputs, None))
            # The output is each feature's change from the last input minute
            outputs = outputs * (features_std + 1e-8) + x[:, -1:, :]
            preds.append(outputs.cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def r2_vs_baseline(err, baseline_err):
    """Out-of-sample R2: 1 - MSE / baseline MSE. Above 0 means the model beats the baseline."""
    return 1 - np.mean(err ** 2) / np.mean(baseline_err ** 2)


def report(preds, inputs, labels):
    inputs = inputs.astype(np.float64)
    labels = labels[:, :preds.shape[1]].astype(np.float64)
    # Undo the log scale on volume and barCount so every metric is in original units
    for values in (preds, inputs, labels):
        values[..., log_columns] = np.expm1(values[..., log_columns])
    last = inputs[:, -1:]

    # Baseline: every future price equals the last input close (a minute's open is
    # almost always the previous close), and volume and barCount equal their average
    # over the input window
    baseline = np.repeat(last[:, :, close_col:close_col + 1], num_cols, axis=2)
    baseline[:, :, log_columns] = inputs[:, :, log_columns].mean(axis=1, keepdims=True)
    baseline = np.broadcast_to(baseline, labels.shape)
    err = preds - labels
    baseline_err = baseline - labels

    print("\nPer feature, over every predicted minute (original units: dollars, shares, trades)")
    print(f"{'feature':<10}{'MAE':>12}{'RMSE':>12}{'base MAE':>12}{'base RMSE':>12}{'R2 vs base':>12}")
    for i, col in enumerate(columns):
        e, b = err[:, :, i], baseline_err[:, :, i]
        print(f"{col:<10}{np.abs(e).mean():>12.6g}{np.sqrt(np.mean(e ** 2)):>12.6g}"
              f"{np.abs(b).mean():>12.6g}{np.sqrt(np.mean(b ** 2)):>12.6g}{r2_vs_baseline(e, b):>12.4f}")

    print("\nClose price by minute ahead. Direction = did the model call the move from the last")
    print("input close up or down correctly (windows where the close didn't move are skipped).")
    print(f"{'minute':<8}{'MAE':>10}{'base MAE':>10}{'R2 vs base':>12}{'direction':>11}{'always up':>11}")
    for h in range(preds.shape[1]):
        actual_move = labels[:, h, close_col] - last[:, 0, close_col]
        pred_move = preds[:, h, close_col] - last[:, 0, close_col]
        moved = actual_move != 0
        direction_acc = np.mean(np.sign(pred_move[moved]) == np.sign(actual_move[moved]))
        always_up_acc = np.mean(actual_move[moved] > 0)
        e, b = err[:, h, close_col], baseline_err[:, h, close_col]
        print(f"{h + 1:<8}{np.abs(e).mean():>10.4f}{np.abs(b).mean():>10.4f}{r2_vs_baseline(e, b):>12.4f}"
              f"{direction_acc:>11.2%}{always_up_acc:>11.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained models on the test set.")
    parser.add_argument("--model", choices=["ten_minute", "one_out", "both"], default="both")
    parser.add_argument("--weights", type=Path,
                        help="A .pth or training checkpoint to evaluate instead of the one in trained_models/. "
                             "Needs --model ten_minute or one_out.")
    args = parser.parse_args()
    if args.weights and args.model == "both":
        parser.error("--weights needs --model ten_minute or --model one_out")
    names = ["ten_minute", "one_out"] if args.model == "both" else [args.model]

    inputs, labels, window_times, input_hours = load_windows()

    # Same chronological split as the training scripts: the test set is the last 20%
    # of windows, after a gap of 2 * input_days so it shares no minutes with training
    test_start = int(len(inputs) * 0.8) + 2 * input_days
    inputs, labels, window_times = inputs[test_start:], labels[test_start:], window_times[test_start:]
    input_hours = input_hours[test_start:]

    print(device)
    print(f"Test samples: {len(inputs)} ({window_times[0]} to {window_times[-1]})")

    for name in names:
        weights_path = args.weights or default_weights[name]
        print(f"\n===== {name} model: {weights_path} =====")
        if not weights_path.exists():
            print("Not found, skipping. Train it first, or pass --weights.")
            continue
        model = load_model(name, weights_path)
        report(predict(model, inputs, input_hours), inputs, labels)


if __name__ == "__main__":
    main()
