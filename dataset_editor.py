import pandas as pd
import numpy as np

input_days = 10
df = pd.read_csv("Stock_Transformer/Data/oneMinData/1_min_SPY_2008-2021.csv")
num_cols = 7
df = df[['open', 'high', 'low', 'close', 'volume', 'barCount', 'average']]

# Calculate the means and standard deviations for each column using only the
# first 80% of rows (the training period), so no test-period data leaks into the scaling
train_rows = int(len(df) * 0.8)
means = df.iloc[:train_rows].mean()
stds = df.iloc[:train_rows].std()

# Standardize the DataFrame
standardized_df = (df - means) / stds

# Create arrays for data chunks and means/stds
array_shape = (standardized_df.shape[0] - input_days + 1, input_days, num_cols)
data_array = np.empty(array_shape)
means_and_devs_array = np.array([means.values, stds.values])

# Loop through df to get chunks of data
for i in range(array_shape[0]):
    data_array[i] = standardized_df.iloc[i:i + input_days].values

# Write the standardized data to a file
with open("Stock_Transformer/Data/oneMinData/standardized_data.csv", 'w') as data_file:
    for window in data_array:
        np.savetxt(data_file, window, delimiter=",", fmt='%f')

# Write the means and standard deviations to a file
with open("Stock_Transformer/Data/oneMinData/means_stds.csv", 'w') as stats_file:
    np.savetxt(stats_file, means_and_devs_array, delimiter=",", fmt='%f')
