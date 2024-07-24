import pandas as pd
import os
import csv
import ast


def create_instance(cluster_id, cluster_data, depot_mapped_id, df, distance_matrix):
    cluster_data_with_depot = [depot_mapped_id] + [plz for plz in cluster_data]
    print(f"Processing cluster {cluster_id} with data: {cluster_data_with_depot}")

    data_df_all = pd.DataFrame()
    for plz in cluster_data_with_depot:
        if plz == depot_mapped_id:
            df_filtered = df[df['MappedId'] == depot_mapped_id]
        else:
            df_filtered = df[df['PLZ'] == plz]

        data_df = df_filtered[['PLZ', 'MappedId', 'Von1', 'Bis1', 'Latitude', 'Longitude']]

        if data_df.empty:
            print(f"No addresses found for PLZ {plz} in cluster {cluster_id}")
        else:
            data_df_all = pd.concat([data_df_all, data_df])

    if data_df_all.empty:
        print(f"No addresses found for any PLZ in cluster {cluster_id}")
    else:
        mapped_ids = data_df_all['MappedId'].tolist()
        print(f"Mapped IDs for cluster {cluster_id}: {mapped_ids}")

        # Ensure the distance matrix is filtered and indexed correctly
        distance_matrix_filtered = distance_matrix.loc[mapped_ids, mapped_ids]

        # Set the index and columns names to mapped_ids
        distance_matrix_filtered.index = mapped_ids
        distance_matrix_filtered.columns = mapped_ids

        # Debugging: Print the shape of the filtered distance matrix
        print(f"Filtered distance matrix shape for cluster {cluster_id}: {distance_matrix_filtered.shape}")

        if distance_matrix_filtered.shape[0] != distance_matrix_filtered.shape[1]:
            print(f"Filtered distance matrix for cluster {cluster_id} is not square: {distance_matrix_filtered.shape}")
        else:
            instance_path = os.path.join('Data/Clusters', f'cluster_{cluster_id}')
            if not os.path.exists(instance_path):
                os.makedirs(instance_path)

            data_df_all.to_csv(os.path.join(instance_path, f'data_{cluster_id}.csv'), index=False)
            distance_matrix_filtered.to_csv(os.path.join(instance_path, f'distance_matrix_{cluster_id}.csv'))
            print(f'Instance {cluster_id} created with {len(data_df_all)} address IDs.')

def main():
    # Load the Excel file into a DataFrame
    df = pd.read_excel('Data/rawData/filtered_shipments_entries_with_mapped_ids.xlsx')
    final_distance_matrix_new = pd.read_excel("Data/rawData/final_distance_matrix_new.xlsx", index_col=0)

    # Extract the shipment entry with address id=1
    depot_mapped_id = 1

    # Load clusters from the CSV file
    clusters_file_path = 'Data/preprocessedData/training_data.csv'
    clusters = []

    with open(clusters_file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header line
        for i, line in enumerate(reader):
            if i >= 1000:  # Stop after reading 1000 clusters
                break
            cluster = ast.literal_eval(line[0])
            clusters.append(cluster)

    # Process each cluster and create instances
    for i, cluster in enumerate(clusters, start=1):
        create_instance(i, cluster, depot_mapped_id, df, final_distance_matrix_new)

if __name__ == "__main__":
    main()
