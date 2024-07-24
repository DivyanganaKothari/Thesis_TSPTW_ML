import os
import shutil
import random
import pandas as pd

# Define the source and target directories
source_dir = './Data/Clusters'
training_dir = './Data/Training_data_1'
test_dir = './Data/Testing_data_1'

# Create target directories if they don't exist
os.makedirs(training_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all clusters in the source directory
all_clusters = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]


# Function to check if a cluster has less than 70 nodes
def has_less_than_70_nodes(cluster):
    data_path = os.path.join(source_dir, cluster, f'data_{cluster.split("_")[-1]}.csv')
    if os.path.exists(data_path):
        data_df = pd.read_csv(data_path)
        return len(data_df) < 70
    return False


# Filter clusters to only include those with less than 70 nodes
filtered_clusters = [cluster for cluster in all_clusters if has_less_than_70_nodes(cluster)]

# Shuffle the list to randomize the selection
random.shuffle(all_clusters)

# Select 200 clusters for training and 100 clusters for testing
training_clusters = all_clusters[:250]
test_clusters = all_clusters[250:350]


# Function to copy selected clusters
def copy_clusters(clusters, target_dir):
    for cluster in clusters:
        src_path = os.path.join(source_dir, cluster)
        dst_path = os.path.join(target_dir, cluster)
        shutil.copytree(src_path, dst_path)


# Copy the selected clusters to the respective target directories
copy_clusters(training_clusters, training_dir)
copy_clusters(test_clusters, test_dir)

print(f"Copied {len(training_clusters)} clusters to {training_dir}")
print(f"Copied {len(test_clusters)} clusters to {test_dir}")
