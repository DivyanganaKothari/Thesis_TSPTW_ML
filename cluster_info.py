import json
import pandas as pd
import os


def create_instance(cluster_id, cluster_data):
    # Limit the number of address IDs to 50
    cluster_data = cluster_data[:50]

    # Load the Excel file into a DataFrame
    df = pd.read_excel('excel_sheets/filtered_shipments_entries_with_mapped_ids.xlsx')
    final_distance_matrix_new = pd.read_excel("excel_sheets/final_distance_matrix_new.xlsx", index_col=0)

    # Extract the shipment entry with address id=1
    depot = df.loc[df['MappedId'] == 1]

    # Add depot to the beginning and end of each cluster
    cluster_data.insert(0, depot['PLZ'].values[0])
    cluster_data.append(depot['MappedId'].values[0])

    # Filter the DataFrame
    df_filtered = df[df['PLZ'].isin(cluster_data)]

    # Extract the required columns
    data_df = df_filtered[['PLZ', 'MappedId', 'Von1', 'Bis1', 'Latitude', 'Longitude']]
    # Filter the distance matrix
    mapped_ids = df_filtered['MappedId'].tolist()
    distance_matrix_filtered = final_distance_matrix_new.loc[mapped_ids, mapped_ids]

    # Create instance directory if it doesn't exist
    instance_path = os.path.join('clusters', f'cluster_{cluster_id}')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    # Save DataFrame to CSV file
    data_df.to_csv(os.path.join(instance_path, f'data_{cluster_id}.csv'), index=False)

    # Save the filtered distance matrix to a new CSV file
    distance_matrix_filtered.to_csv(os.path.join(instance_path, f'distance_matrix_{cluster_id}.csv'), index=False)

    print(f'Instance {cluster_id} created with {len(cluster_data)} address IDs.')


# Load clusters from the JSON file
with open('excel_sheets/clusters.json', 'r') as f:
    clusters = json.load(f)

# Create instances for each cluster
for i, cluster in enumerate(clusters, start=1):
    create_instance(i, cluster)
