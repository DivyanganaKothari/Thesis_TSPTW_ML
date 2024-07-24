import os
import shutil
import random
import pandas as pd

# Define the source and target directories
source_dir = '../Data/Clusters'
training_dir = '../Data/Training_data_2'
test_dir = '../Data/Testing_data_2'

# Create target directories if they don't exist
os.makedirs(training_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
# Clear the target directories before copying
for dir_path in [training_dir, test_dir]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# List all clusters in the source directory
all_clusters = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

filtered_clusters = [cluster for cluster in all_clusters]

# Shuffle the list to randomize the selection
random.shuffle(filtered_clusters)

# Specify the starting index and the number of clusters for the new sets
start_index = 350  # Update this value based on the previous selection
num_training_clusters = 350
num_test_clusters = 150

# Select clusters for training and testing starting from the specified index
training_clusters = filtered_clusters[start_index:start_index + num_training_clusters]
test_clusters = filtered_clusters[start_index + num_training_clusters:start_index + num_training_clusters + num_test_clusters]

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
