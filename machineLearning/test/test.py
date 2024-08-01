import pandas as pd
import numpy as np
import os
import ast

# Load the filtered shipment entries
filtered_shipment_entries = pd.read_csv('../../Data/preprocessedData/filtered_shipment_entries.csv')

# Load the distance matrix
distance_matrix = pd.read_csv('../../Data/preprocessedData/filtered_distance_matrix.csv', index_col=0)

# Ensure AddressId columns are treated as integers
distance_matrix.index = distance_matrix.index.astype(int)
distance_matrix.columns = distance_matrix.columns.astype(int)

# Load the zip codes and tour length CSV
zip_codes_df = pd.read_csv('../../Data/zip_tourLength/optimized_511_2024_04_02_clusters_20.csv', sep=';')

# Parse the zip codes column to convert string representation of lists into actual lists
zip_codes_df['Zip Codes'] = zip_codes_df['Zip Codes'].apply(lambda x: ast.literal_eval(x.strip()))

# Load the depot node information
depot_node_info = pd.DataFrame([{
    'SdnId': 21114458,
    'KdnId': -22235,
    'AddressId': 1,
    'StopZeit': 184,
    'Von1': '2024-04-04 00:00:00',
    'Bis1': '2024-04-04 23:59:59',
    'Latitude': 47.52725,
    'Longitude': 7.6854
}])

# Ensure depot AddressId is added to the distance matrix
if 1 not in distance_matrix.index:
    depot_distances = pd.Series(0, index=distance_matrix.columns)
    distance_matrix.loc[1] = depot_distances
    distance_matrix[1] = depot_distances

# Directory to save the extracted information
extracted_info_dir = '../../Data/TestInputFeaturesCheck7'
if not os.path.exists(extracted_info_dir):
    os.makedirs(extracted_info_dir)

# Function to calculate time window features
def calculate_time_window_features(filtered_info):
    filtered_info['Von1'] = pd.to_datetime(filtered_info['Von1'], format='%Y-%m-%d %H:%M:%S')
    filtered_info['Bis1'] = pd.to_datetime(filtered_info['Bis1'], format='%Y-%m-%d %H:%M:%S')

    mask = ~((filtered_info['Von1'] == pd.Timestamp('2024-04-04 00:00:00')) &
             (filtered_info['Bis1'] == pd.Timestamp('2024-04-04 23:59:59')))
    calculation_info = filtered_info[mask]

    reference_time = pd.Timestamp('2024-04-04 00:00:00')

    time_windows = [(row['Von1'], row['Bis1']) for _, row in calculation_info.iterrows()]
    if len(time_windows) == 0:
        return {
            'Total Time Window': -1,
            'Average Time Window': -1,
            'Standard Deviation of Time Window': -1,
            'Average Earliest Time': -1,
            'Standard Deviation Earliest Time': -1,
            'Average Latest Time': -1,
            'Standard Deviation Latest Time': -1,
            'Mean Time Window': -1
        }

    total_time_window = sum([(end - start).total_seconds() / 60 for start, end in time_windows])
    average_time_window = total_time_window / len(time_windows)
    std_dev_time_window = np.std([(end - start).total_seconds() / 60 for start, end in time_windows])

    earliest_times = [(tw[0] - reference_time).total_seconds() / 60 for tw in time_windows]
    min_earliest_time = np.min(earliest_times)
    max_earliest_time = np.max(earliest_times)
    average_earliest_time = np.mean(earliest_times)
    std_dev_earliest_time = np.std(earliest_times)

    latest_times = [(tw[1] - reference_time).total_seconds() / 60 for tw in time_windows]
    min_latest_time = np.min(latest_times)
    max_latest_time = np.max(latest_times)
    average_latest_time = np.mean(latest_times)
    std_dev_latest_time = np.std(latest_times)

    mean_time_window = np.mean([end - start for start, end in time_windows])

    return {
        'Total Time Window': total_time_window,
        'Average Time Window': average_time_window,
        'Standard Deviation of Time Window': std_dev_time_window,
        'Average Earliest Time': average_earliest_time,
        'Standard Deviation Earliest Time': std_dev_earliest_time,
        'Average Latest Time': average_latest_time,
        'Standard Deviation Latest Time': std_dev_latest_time,
        'Min Earliest Time': min_earliest_time,
        'Max Earliest Time': max_earliest_time,
        'Min Latest Time': min_latest_time,
        'Max Latest Time': max_latest_time,
        'Mean Time Window': mean_time_window.total_seconds() / 60
    }

# Function to calculate distance features
def calculate_distance_features(filtered_info, distance_matrix):
    address_ids = filtered_info['AddressId'].astype(int).tolist()
    if 1 not in address_ids:
        address_ids.append(1)

    distance_matrix_cluster = distance_matrix.loc[address_ids, address_ids]
    n = distance_matrix_cluster.shape[0]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    all_distances = distance_matrix_cluster.values[mask]

    avg_node = np.mean(distance_matrix_cluster.values, axis=0)
    euclidean_distances_to_avg_node = np.linalg.norm(distance_matrix_cluster.values - avg_node[:, np.newaxis], axis=1)

    nearest_neighbors = np.min(distance_matrix_cluster.values + np.eye(n) * np.max(distance_matrix_cluster.values), axis=1)
    farthest_neighbors = np.max(distance_matrix_cluster.values, axis=1)

    return {
        'Total Number of Nodes': n,
        'MinP': np.min(all_distances),
        'MaxP': np.max(all_distances),
        'VarP': np.var(all_distances),
        'SumMinP': np.sum(nearest_neighbors),
        'SumMaxP': np.sum(farthest_neighbors),
        'MinM': np.min(euclidean_distances_to_avg_node),
        'MaxM': np.max(euclidean_distances_to_avg_node),
        'SumM': np.sum(euclidean_distances_to_avg_node),
        'VarM': np.var(euclidean_distances_to_avg_node),
        'Mean Distance': np.mean(all_distances),
        'Median Distance': np.median(all_distances),
        'Std Distance': np.std(all_distances),
        'Sum Distance': np.sum(all_distances),
        'Sum of Min Distance': np.sum(np.min(all_distances)),
        'Sum of Max Distance': np.sum(np.max(all_distances)),
        'Percentile 25 Distance': np.percentile(all_distances, 25),
        'Percentile 50 Distance': np.percentile(all_distances, 50),
        'Percentile 75 Distance': np.percentile(all_distances, 75)
    }

# Function to calculate depot-related features
def calculate_depot_features(filtered_info, distance_matrix):
    depot_distances = distance_matrix.loc[1, filtered_info['AddressId'].astype(int).tolist()].values
    depot_distances_excluding_self = depot_distances[depot_distances != 0]

    return {
        'Sum of Distance to Depot': np.sum(depot_distances_excluding_self),
        'Average Distance to Depot': np.mean(depot_distances_excluding_self),
        'Maximum Distance to Depot': np.max(depot_distances_excluding_self),
        'Minimum Distance to Depot': np.min(depot_distances_excluding_self),
        'Standard Deviation of Distance to Depot': np.std(depot_distances_excluding_self)
    }

# Function to calculate the input features
def calculate_input_features(filtered_info, distance_matrix, cluster_id, tour_length):
    distance_features = calculate_distance_features(filtered_info, distance_matrix)
    time_window_features = calculate_time_window_features(filtered_info)
    depot_features = calculate_depot_features(filtered_info, distance_matrix)

    features = {
        **distance_features,
        **time_window_features,
        **depot_features,
        'Tour Length': tour_length
    }

    return features

# Initialize a list to store the summary information
summary_info = []

# Iterate over each row in the zip codes dataframe
for index, row in zip_codes_df.iterrows():
    cluster_id = index + 1  # Assign a unique cluster ID
    zip_codes = row['Zip Codes']
    tour_length = row['Tour Length [min]']

    if tour_length == -1:
        print(f"Skipping cluster {cluster_id} with zip codes {zip_codes} due to invalid tour length.")
        continue

    # Filter the shipment entries based on the zip codes
    filtered_info = filtered_shipment_entries[filtered_shipment_entries['PLZ'].isin(zip_codes)]

    if not filtered_info.empty:
        # Add the depot information to the filtered info
        filtered_info = pd.concat([filtered_info, depot_node_info], ignore_index=True)
        # Calculate the input features
        try:
            input_features = calculate_input_features(filtered_info, distance_matrix, cluster_id, tour_length)
            summary_info.append(input_features)
        except Exception as e:
            print(f"Error for cluster {cluster_id} with zip codes {zip_codes}: {e}")
    else:
        print(f"No shipment data found for cluster {cluster_id} with zip codes {zip_codes}")

# Create a summary DataFrame
summary_df = pd.DataFrame(summary_info)

# Save the summary information
summary_path = os.path.join(extracted_info_dir, 'input_features_test_20.csv')
summary_df.to_csv(summary_path, index=False)

print("Input features extracted and saved successfully.")
