import pandas as pd
import numpy as np
import os
import csv
import ast


def create_instance(cluster_id, cluster_data, depot_mapped_id, df, distance_matrix):
    cluster_data_with_depot = [depot_mapped_id] + [plz for plz in cluster_data]

    data_df_all = pd.DataFrame()
    for plz in cluster_data_with_depot:
        if plz == depot_mapped_id:
            df_filtered = df[df['MappedId'] == depot_mapped_id]
        else:
            df_filtered = df[df['PLZ'] == plz]

        data_df = df_filtered[['PLZ', 'MappedId', 'Von1', 'Bis1', 'Latitude', 'Longitude']]

        if not data_df.empty:
            data_df_all = pd.concat([data_df_all, data_df])

    if not data_df_all.empty:
        mapped_ids = data_df_all['MappedId'].tolist()

        distance_matrix_filtered = distance_matrix.loc[mapped_ids, mapped_ids]
        distance_matrix_filtered.index = mapped_ids
        distance_matrix_filtered.columns = mapped_ids

        if distance_matrix_filtered.shape[0] == distance_matrix_filtered.shape[1]:
            instance_path = os.path.join('Data/Clusters', f'cluster_{cluster_id}')
            if not os.path.exists(instance_path):
                os.makedirs(instance_path)

            data_df_all.to_csv(os.path.join(instance_path, f'data_{cluster_id}.csv'), index=False)
            distance_matrix_filtered.to_csv(os.path.join(instance_path, f'distance_matrix_{cluster_id}.csv'))


def calculate_distance_features(distance_matrix, nodes):
    n = distance_matrix.shape[0]
    all_distances = distance_matrix.flatten()

    avg_node = np.mean(distance_matrix, axis=0)
    distances_to_avg_node = np.linalg.norm(distance_matrix - avg_node[:, np.newaxis], axis=1)

    nearest_neighbors = np.min(distance_matrix + np.eye(n) * np.max(distance_matrix), axis=1)
    farthest_neighbors = np.max(distance_matrix, axis=1)

    latitudes = [node['Latitude'] for node in nodes]
    longitudes = [node['Longitude'] for node in nodes]

    return {
        'Total number of Nodes': n,
        'MinP': np.min(all_distances),
        'MaxP': np.max(all_distances),
        'VarP': np.var(all_distances),
        'SumMinP': np.sum(nearest_neighbors),
        'SumMaxP': np.sum(farthest_neighbors),
        'MinM': np.min(distances_to_avg_node),
        'MaxM': np.max(distances_to_avg_node),
        'SumM': np.sum(distances_to_avg_node),
        'VarM': np.var(distances_to_avg_node),
        'VarX×VarY': np.var(latitudes) * np.var(longitudes),
        'mean_distance': np.mean(all_distances),
        'median_distance': np.median(all_distances),
        'std_distance': np.std(all_distances),
        'min_distance': np.min(all_distances),
        'max_distance': np.max(all_distances),
        'sum_distance': np.sum(all_distances),
        'density': np.sum(all_distances) / len(distance_matrix),
        'percentile_25': np.percentile(all_distances, 25),
        'percentile_50': np.percentile(all_distances, 50),
        'percentile_75': np.percentile(all_distances, 75)
    }


def calculate_time_window_features(time_windows):
    total_time_window = sum([end - start for start, end in time_windows])
    average_time_window = total_time_window / len(time_windows)
    max_time_window = max([end - start for start, end in time_windows])
    min_time_window = min([end - start for start, end in time_windows])
    std_dev_time_window = np.std([end - start for start, end in time_windows])

    return {
        'Total Time Window': total_time_window,
        'Average Time Window': average_time_window,
        'Max Time Window': max_time_window,
        'Min Time Window': min_time_window,
        'Standard Deviation of Time Window': std_dev_time_window
    }


def calculate_node_features(nodes):
    num_nodes = len(nodes)
    latitudes = [node['Latitude'] for node in nodes]
    longitudes = [node['Longitude'] for node in nodes]
    mean_latitude = np.mean(latitudes)
    mean_longitude = np.mean(longitudes)
    std_dev_latitude = np.std(latitudes)
    std_dev_longitude = np.std(longitudes)

    min_latitude = np.min(latitudes)
    max_latitude = np.max(latitudes)
    min_longitude = np.min(longitudes)
    max_longitude = np.max(longitudes)

    center_latitude = mean_latitude
    center_longitude = mean_longitude
    distances_from_center = [np.sqrt((lat - center_latitude) ** 2 + (lon - center_longitude) ** 2) for lat, lon in
                             zip(latitudes, longitudes)]
    avg_distance_from_center = np.mean(distances_from_center)
    max_distance_from_center = np.max(distances_from_center)
    min_distance_from_center = np.min(distances_from_center)
    std_dev_distance_from_center = np.std(distances_from_center)

    return {
        'Mean Latitude': mean_latitude,
        'Mean Longitude': mean_longitude,
        'Standard Deviation of Latitude': std_dev_latitude,
        'Standard Deviation of Longitude': std_dev_longitude,
        'Min Latitude': min_latitude,
        'Max Latitude': max_latitude,
        'Min Longitude': min_longitude,
        'Max Longitude': max_longitude,
        'Average Distance from Center': avg_distance_from_center,
        'Max Distance from Center': max_distance_from_center,
        'Min Distance from Center': min_distance_from_center,
        'Standard Deviation of Distance from Center': std_dev_distance_from_center
    }


def calculate_depot_features(depot, nodes):
    depot_lat, depot_lon = depot['Latitude'], depot['Longitude']
    distances_from_depot = [np.sqrt((node['Latitude'] - depot_lat) ** 2 + (node['Longitude'] - depot_lon) ** 2) for node
                            in nodes]

    avg_distance_from_depot = np.mean(distances_from_depot)
    max_distance_from_depot = np.max(distances_from_depot)
    min_distance_from_depot = np.min(distances_from_depot)
    std_dev_distance_from_depot = np.std(distances_from_depot)

    return {
        'AvgDepotDist': avg_distance_from_depot,
        'MaxDepotDist': max_distance_from_depot,
        'MinDepotDist': min_distance_from_depot,
        'StdDevDepotDist': std_dev_distance_from_depot
    }


def extract_features(cluster_dir, is_training=True):
    cluster_id = int(cluster_dir.split('_')[-1])

    shipment_data_path = os.path.join(cluster_dir, f'data_{cluster_id}.csv')
    shipment_data = pd.read_csv(shipment_data_path)

    distance_matrix_path = os.path.join(cluster_dir, f'distance_matrix_{cluster_id}.csv')
    distance_matrix_df = pd.read_csv(distance_matrix_path, header=0, index_col=0)
    distance_matrix = distance_matrix_df.values

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"Distance matrix for cluster {cluster_id} is not square: {distance_matrix.shape}")

    nodes = shipment_data.to_dict('records')

    # Separate the depot and other nodes
    depot = nodes[0]  # Assuming the first node is always the depot
    other_nodes = nodes[1:]

    distance_features = calculate_distance_features(distance_matrix, nodes)
    time_window_features = calculate_time_window_features([(node['Von1'], node['Bis1']) for node in nodes])
    node_features = calculate_node_features(nodes)
    depot_features = calculate_depot_features(depot, other_nodes)

    features = {**distance_features, **time_window_features, **node_features, **depot_features}

    if is_training:
        solution_path = os.path.join(cluster_dir, f'solution_{cluster_id}.csv')
        try:
            solution_df = pd.read_csv(solution_path)
            if 'Time_of_the_route' in solution_df.columns:
                time_route_str = solution_df['Time_of_the_route'].iloc[0]
                tour_length = int(time_route_str.split(': ')[1].replace('sec', ''))
            else:
                raise ValueError(f"'Time_of_the_route' column not found in solution file for cluster {cluster_id}")
        except FileNotFoundError:
            tour_length = -1
        except Exception as e:
            tour_length = -1

        if tour_length == 0:
            tour_length = -1

        features['tour_length'] = tour_length

    return features


def process_clusters(clusters_dir, results_file, is_training=True):
    cluster_dirs = [os.path.join(clusters_dir, d) for d in os.listdir(clusters_dir) if
                    os.path.isdir(os.path.join(clusters_dir, d))]

    all_features = []
    for cluster_dir in cluster_dirs:
        try:
            features = extract_features(cluster_dir, is_training)
            all_features.append(features)
        except ValueError as e:
            print(e)

    if not all_features:
        print("No features were extracted. Check for errors in processing clusters.")
    else:
        features_df = pd.DataFrame(all_features)
        features_df.to_csv(results_file, index=False)


def main():
    training_clusters_dir = '../Data/Training_data_2'
    testing_clusters_dir = '../Data/Testing_data_2'
    results_dir = '../Data/Results'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Process training data
    process_clusters(training_clusters_dir, os.path.join(results_dir, 'Training_input_features_2.csv'), is_training=True)

    # Process testing data
    process_clusters(testing_clusters_dir, os.path.join(results_dir, 'Testing_input_features_2.csv'), is_training=False)


if __name__ == "__main__":
    main()
