import os
import pandas as pd
import ast
import numpy as np


def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    max_distance = float('-inf')
    min_distance = float('inf')
    distances = [] # List to store the distances in the route
    for i in range(len(route) - 1):
        from_node = route[i] - 1
        to_node = route[i + 1] - 1
        current_distance = distance_matrix[from_node][to_node]
        distances.append(current_distance)  # Add the current distance to the list
        total_distance += current_distance
        max_distance = max(max_distance, current_distance)
        min_distance = min(min_distance, current_distance)
    std_dev_distance = np.std(distances)
    average_distance = total_distance / (len(route) - 1)

    return total_distance, average_distance, max_distance, min_distance, std_dev_distance


# Define the path to the clusters directory
clusters_dir = 'clusters'
# Get a list of all cluster directories
cluster_dirs = [d for d in os.listdir(clusters_dir) if os.path.isdir(os.path.join(clusters_dir, d))]

# extract route sequence, total distance, total nodes, coordinates,time window for each tour and write to a file

with open('tour_output.txt', 'w') as f:
    # Iterate over each cluster directory
    for cluster_dir in cluster_dirs:
        # load the distance matrix
        distance_matrix_path = os.path.join(clusters_dir, cluster_dir,
                                            f'distance_matrix_{cluster_dir.split("_")[-1]}.csv')
        distance_matrix_df = pd.read_csv(distance_matrix_path, index_col=0)

        # Convert the DataFrame to a numpy array
        distance_matrix = distance_matrix_df.values
        # Load the solution file
        solution_path = os.path.join(clusters_dir, cluster_dir, f'solution_{cluster_dir.split("_")[-1]}.csv')

        try:
            solution_df = pd.read_csv(solution_path)
        except FileNotFoundError:
            print(f"Solution file not found for cluster {cluster_dir}")
            continue

        # Iterate over each row in the DataFrame
        for index, row in solution_df.iterrows():
            # Convert the route string into a list of dictionaries
            route_dicts = ast.literal_eval(row['route'])
            # Extract the 'node' values from each dictionary to get the route as a list of integers
            route = [node_dict['node'] for node_dict in route_dicts]
            # Call the calculate_total_distance function for the route

            total_distance, average_distance, max_distance, min_distance, standard_deviation_distance = calculate_total_distance(route,
                                                                                                    distance_matrix)

            # Extract the 'coordinate' values from each dictionary to get the coordinates as a list of tuples
            coordinates = [node_dict['coordinate'] for node_dict in route_dicts]

            # Extract the 'time_min' and 'time_max' values from each dictionary to get the time window as a list of tuples
            time_window = [(node_dict['time_min'], node_dict['time_max']) for node_dict in route_dicts]

            # Get the total number of nodes in the tour
            total_nodes = len(route)

            # Write the tour information to the output file
            f.write(f"Cluster: {cluster_dir.split('_')[-1]}\n")
            f.write(f"Route: {route}\n")
            f.write(f"Coordinates: {coordinates}\n")
            f.write(f"Time Window: {time_window}\n")
            f.write(f"Total Nodes: {total_nodes}\n")
            f.write(f"Total Distance: {total_distance}\n")
            f.write(f"Average Distance: {average_distance}\n")
            f.write(f"Max Distance: {max_distance}\n")
            f.write(f"Min Distance: {min_distance}\n")
            f.write(f"Standard Deviation of Distance: {standard_deviation_distance}\n")
            f.write("\n")
            print(
                f"Cluster_{cluster_dir.split('_')[-1]}, total_node: {total_nodes}, total_distance: {total_distance}, average_distance: {average_distance}, maximum_distance : {max_distance}, minimum_distance : {min_distance}, Standard Deviation of Distance : {standard_deviation_distance}, route: {route}, coordinates: {coordinates}, time window: {time_window}")

# depot_node,txt contains depot address id
# Load the data file for the first cluster
first_cluster_dir = cluster_dirs[0]
data_path = os.path.join(clusters_dir, first_cluster_dir, f'data_{first_cluster_dir.split("_")[-1]}.csv')

try:
    data_df = pd.read_csv(data_path)

    # The depot node is the first node in the data file
    depot_node = data_df.iloc[0].to_dict()

    # Print the depot node
    print(f"Depot node: {depot_node}")

    # Save the depot node to a file
    with open('depot_node.txt', 'w') as f:
        f.write(str(depot_node))

except FileNotFoundError:
    print(f"Data file not found for cluster {first_cluster_dir}")


# saving all input features to dataframe

clusters_dir = 'clusters'

tours = []

with open('tour_output.txt', 'r') as f:
    lines = f.readlines()
    tour = {}
    for i, line in enumerate(lines):
        line = line.strip()  # Strip leading and trailing whitespace
        if line.startswith('Cluster'):
            # Save the previous tour data
            if tour:
                tours.append(tour)
                # Start a new tour
            tour = {'Cluster': int(line.split(':')[1].strip())}
            # Adding the path to the distance matrix file for the cluster
            tour['Distance Matrix Path'] = os.path.join(clusters_dir, f'cluster_{tour["Cluster"]}', f'distance_matrix_{tour["Cluster"]}.csv')

        elif line.startswith('Route:'):
                tour['Route'] = eval(line.split(':')[1].strip())
        elif line.startswith('Coordinates:'):
                tour['Coordinates'] = eval(line.split(':')[1].strip())
        elif line.startswith('Time Window:'):
                tour['Time Window'] = eval(line.split(':')[1].strip())
        elif line.startswith('Total Nodes:'):
                tour['Total Nodes'] = int(line.split(':')[1].strip())
        elif line.startswith('Total Distance:'):
                tour['Total Distance'] = float(line.split(':')[1].strip())
        elif line.startswith('Average Distance:'):
                tour['Average Distance'] = float(line.split(':')[1].strip())
        elif line.startswith('Max Distance:'):
                tour['Max Distance'] = float(line.split(':')[1].strip())
        elif line.startswith('Min Distance:'):
                tour['Min Distance'] = float(line.split(':')[1].strip())
        elif line.startswith('Standard Deviation of Distance:'):
                tour['Standard Deviation of Distance'] = float(line.split(':')[1].strip())
            # Save the last tour data
    if tour:
        tours.append(tour)

# Read the depot_node.txt file
with open('depot_node.txt', 'r') as f:
    depot_node = eval(f.read())

# Add the depot node information to each tour
for tour in tours:
     tour['Depot Node'] = depot_node

# Convert the list of tours into a pandas DataFrame
features_df = pd.DataFrame(tours)

#  display option to display all columns
# pd.set_option('display.max_columns', None)  # None means unlimited
# pd.set_option('display.expand_frame_repr', False)  # Prevent line-wrapping
# pd.set_option('display.max_colwidth', None)  # None means unlimited
print(features_df)


# Assuming features is a list of dictionaries where
# each dictionary represents an instance and
# each key-value pair is a feature name and its value.
