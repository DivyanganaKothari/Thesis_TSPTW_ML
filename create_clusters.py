import pandas as pd
import ast
import itertools
import pandas as pd
import ast

# Load the Excel file into a DataFrame
df_shipments = pd.read_excel('Data/rawData/filtered_shipments_entries_with_mapped_ids.xlsx')

# Extract the unique PLZ values
shipment_plz = df_shipments['PLZ'].unique()

# Read the CSV file containing zip codes and their neighbors
file_path = 'Data/preprocessedData/filtered_common_zip_code_neighbors.csv'
zip_data = pd.read_csv(file_path)
zip_data['neighbors'] = zip_data['neighbors'].apply(ast.literal_eval)

# Filter the neighbors to only include those that are present in the shipment entries
zip_data['neighbors'] = zip_data['neighbors'].apply(lambda neighbors: [plz for plz in neighbors if plz in shipment_plz])

# Save the filtered zip data to a new CSV file
output_file_path = 'Data/preprocessedData/filtered_common_zip_code_neighbors_filtered.csv'
zip_data.to_csv(output_file_path, index=False)

print(f"Filtered zip data saved to {output_file_path}")


# Read the CSV file into a DataFrame
df = pd.read_csv('Data/preprocessedData/filtered_common_zip_code_neighbors_filtered.csv')

# Convert the string representation of the list to an actual list
df['neighbors'] = df['neighbors'].apply(ast.literal_eval)

# Convert each element in the list to an integer
df['neighbors'] = df['neighbors'].apply(lambda x: [int(i) for i in x])

# Save the DataFrame to a new CSV file
df.to_csv('Data/preprocessedData/filtered_common_zip_code_neighbors_filtered_int.csv', index=False)

# Read the CSV file containing zip codes and their neighbors
file_path = 'Data/preprocessedData/filtered_common_zip_code_neighbors_filtered_int.csv'
zip_data = pd.read_csv(file_path)
zip_data['neighbors'] = zip_data['neighbors'].apply(ast.literal_eval)

# Create a dictionary of zip codes and their neighbors
neighbors_dict = dict(zip(zip_data['zip_code'], zip_data['neighbors']))

# Extract individual zip codes
individual_zip_codes = list(neighbors_dict.keys())

# Function to generate clusters from a zip code and its neighbors
def generate_clusters(zip_code, neighbors_dict):
    clusters = []
    # Generate clusters of size 1 (individual zip code)
    clusters.append([zip_code])
    # Generate clusters of size 2, 3, ... including neighbors
    for size in range(2, len(neighbors_dict[zip_code]) + 2):
        for combination in itertools.combinations([zip_code] + neighbors_dict[zip_code], size):
            clusters.append(list(combination))
    return clusters

# Generate all possible clusters
all_clusters = []
for zip_code in neighbors_dict.keys():
    all_clusters.extend(generate_clusters(zip_code, neighbors_dict))

# Remove duplicate clusters (since they can be generated in different orders)
all_clusters = [list(cluster) for cluster in set(tuple(sorted(cluster)) for cluster in all_clusters)]


# Append individual zip code clusters to all_clusters
all_clusters.extend([[zip_code] for zip_code in individual_zip_codes])

print(f"Generated {len(all_clusters)} unique clusters from the provided data.")

# Convert the list of clusters to a DataFrame
clusters_df = pd.DataFrame({'Cluster': [str(cluster) for cluster in all_clusters]})

# Save the clusters to a CSV file
output_file_path = 'Data/preprocessedData/clusters_list.csv'
clusters_df.to_csv(output_file_path, index=False)

print(f"Clusters saved to {output_file_path}")
