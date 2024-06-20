import pandas as pd
import numpy as np

input_days = 10
df = pd.read_csv("Stock_Transformer/Data/oneMinData/1_min_SPY_2008-2021.csv")
num_cols = 7
df = df[['open', 'high', 'low', 'close', 'volume', 'barCount', 'average']]
# make all the arrays
array_shape = (df.shape[0]-input_days+1, input_days, num_cols)
array_shape_w_standard = (df.shape[0]-input_days+1, num_cols, 2)
data_array = np.empty(array_shape)
means_and_dev_data_array = np.empty(array_shape_w_standard)
ans_array = np.empty(array_shape)
means_and_dev_ans_array = np.empty(array_shape_w_standard)
# loop through df to get chunks of data
for i in range(array_shape[0]-(2*input_days)):
    chunk = df.iloc[i:i + input_days]  # Get the chunk of data
    arr = chunk.to_numpy()  # Convert to numpy array
    ans = df.iloc[i+input_days:i+(2*input_days)]
    ans = ans.to_numpy()

    # Standardize the data
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    standardized_data = (arr - means) / stds
    means2 = ans.mean(axis=0)
    stds2 = ans.std(axis=0)
    standardized_data2 = (ans - means2) / stds2

    # Store in data arrays
    data_array[i] = standardized_data
    ans_array[i] = standardized_data2

    # Store means and standard deviations
    for j in range(num_cols):
        means_and_dev_data_array[i, j, 0] = means[j]
        means_and_dev_data_array[i, j, 1] = stds[j]
        means_and_dev_ans_array[i, j, 0] = means2[j]
        means_and_dev_ans_array[i, j, 1] = stds2[j]
    print(str(i) + "/" + str(array_shape[0]-input_days+1))

# should work
df3 = pd.DataFrame(means_and_dev_data_array.reshape(-1, num_cols*2), columns = [f'Mean' if i%2==0 else f'Standard_Deviation' for i in range(num_cols*2)])
df4 = pd.DataFrame(means_and_dev_ans_array.reshape(-1, num_cols*2), columns = [f'Mean' if i%2==0 else f'Standard_Deviation' for i in range(num_cols*2)])
df3.to_csv('train_avgs.csv')
df4.to_csv('answers_avgs.csv')
'''
# Flatten each 2D slice (each instance) into a single row
flattened_data = data_array.reshape(-1, input_days * num_cols)
flattened_data2 = ans_array.reshape(-1, input_days * num_cols)
# Convert the flattened data to a DataFrame
column_names = [f'Day{day+1}_Feature{feature+1}' for day in range(input_days) for feature in range(num_cols)]
df1 = pd.DataFrame(flattened_data, columns=column_names)
column_names = [f'Day{day+1}_Feature{feature+1}' for day in range(input_days) for feature in range(num_cols)]
df2 = pd.DataFrame(flattened_data2, columns=column_names)

# Write the DataFrame to a CSV file
df1.to_csv('train.csv', index=False)
df2.to_csv('answers.csv', index=False)
'''
