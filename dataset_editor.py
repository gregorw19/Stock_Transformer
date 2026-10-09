from pathlib import Path
import pandas as pd
import numpy as np

# Paths are relative to this file, so the scripts work from any working directory
project_dir = Path(__file__).resolve().parent
data_dir = project_dir.parent / "Data"
raw_data_path = data_dir / "oneMinData" / "1_min_SPY_2008-2021.csv"
dataset_path = data_dir / "spy_1min_dataset.npz"

columns = ['open', 'high', 'low', 'close', 'volume', 'barCount', 'average']
log_columns = [columns.index('volume'), columns.index('barCount')]  # Modeled on a log scale
input_days = 10  # minutes of input in each window
label_days = 10  # minutes after the input that the model predicts

# Keep only the regular session. The file's times are two hours behind New York,
# so 07:30-13:59 is 9:30-15:59 ET. Some days also have pre-market and after-hours
# bars, many of them with zero volume.
regular_hours_only = True


def build_dataset():
    df = pd.read_csv(raw_data_path)
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d  %H:%M:%S")

    # The raw file stores its trading days in shuffled order and repeats ~640k rows.
    # Sort and de-duplicate so windows are made of consecutive minutes and the
    # training scripts' chronological train/test split really is chronological.
    df = df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

    if regular_hours_only:
        time_of_day = df['date'].dt.strftime('%H:%M')
        df = df[(time_of_day >= '07:30') & (time_of_day <= '13:59')].reset_index(drop=True)

    bars = df[columns].to_numpy(dtype=np.float32)
    times = df['date'].to_numpy().astype('datetime64[m]')

    # A window is valid only if its input_days + label_days rows are consecutive
    # minutes, so no window spans the gap between two trading days
    window_len = input_days + label_days
    elapsed = times[window_len - 1:] - times[:len(times) - window_len + 1]
    starts = np.flatnonzero(elapsed == np.timedelta64(window_len - 1, 'm'))

    np.savez(dataset_path, bars=bars, times=times, starts=starts)

    print(f"Bars: {len(bars)} from {times[0]} to {times[-1]} ({df['date'].dt.date.nunique()} trading days)")
    print(f"Windows: {len(starts)} ({input_days} input + {label_days} label minutes each)")
    print(f"Saved to {dataset_path}")


def load_windows(path=dataset_path):
    """Loads the dataset written by build_dataset().

    Returns unnormalized arrays, with volume and barCount as log(1 + value):
        inputs:      (num_windows, input_days, 7) the input minutes of each window
        labels:      (num_windows, label_days, 7) the minutes right after them
        start_times: (num_windows,) the first minute of each window
        input_hours: (num_windows, input_days) the time of day of each input minute,
                     in hours in the data file's time zone (07:30 is 7.5)
    Windows are in chronological order.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run dataset_editor.py first.")

    data = np.load(path)
    bars, starts = data['bars'], data['starts']
    # Volume and barCount have rare, huge spikes; a log scale keeps them from dominating
    bars[:, log_columns] = np.log1p(bars[:, log_columns])
    windows = bars[starts[:, None] + np.arange(input_days + label_days)]
    start_times = data['times'][starts]
    start_hours = (start_times - start_times.astype('datetime64[D]')).astype(np.int64) / 60
    input_hours = start_hours[:, None] + np.arange(input_days) / 60
    return windows[:, :input_days], windows[:, input_days:], start_times, input_hours


if __name__ == "__main__":
    build_dataset()
