from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
data = pd.read_csv('../Data/preprocessedData/clusters_list.csv')

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

# Save the training and test sets to CSV files
train_data.to_csv('../Data/preprocessedData/training_data.csv', index=False)
test_data.to_csv('../Data/preprocessedData/test_data.csv', index=False)