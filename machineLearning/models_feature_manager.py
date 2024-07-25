import pandas as pd
import numpy as np
import os


class FeatureManager:
    def __init__(self):
        self.feature_sets = {
            "set1": ['Total number of Nodes', 'MinP', 'MaxP', 'VarP', 'SumMinP', 'SumMaxP', 'MinM', 'MaxM', 'SumM',
                     'VarM', 'VarX×VarY', 'mean_distance', 'median_distance', 'std_distance', 'min_distance',
                     'max_distance', 'sum_distance', 'density', 'percentile_25', 'percentile_50', 'percentile_75',
                     'Total Time Window', 'Average Time Window', 'Max Time Window', 'Min Time Window',
                     'Standard Deviation of Time Window', 'Mean Latitude', 'Mean Longitude',
                     'Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude', 'Average Distance from Center',
                     'Max Distance from Center', 'Min Distance from Center',
                     'Standard Deviation of Distance from Center', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set2": ['Total number of Nodes', 'MinP', 'MaxP', 'VarP', 'SumMinP', 'SumMaxP', 'MinM', 'MaxM', 'SumM',
                     'VarM', 'VarX×VarY', 'Total Time Window', 'Average Time Window', 'Max Time Window',
                     'Min Time Window', 'Standard Deviation of Time Window', 'Mean Latitude', 'Mean Longitude',
                     'Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude', 'Average Distance from Center',
                     'Max Distance from Center', 'Min Distance from Center',
                     'Standard Deviation of Distance from Center', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set3": ['Total number of Nodes', 'mean_distance', 'median_distance', 'std_distance', 'min_distance',
                     'max_distance', 'sum_distance', 'density', 'percentile_25', 'percentile_50', 'percentile_75',
                     'Total Time Window', 'Average Time Window', 'Max Time Window', 'Min Time Window',
                     'Standard Deviation of Time Window', 'Mean Latitude', 'Mean Longitude',
                     'Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude', 'Average Distance from Center',
                     'Max Distance from Center', 'Min Distance from Center',
                     'Standard Deviation of Distance from Center', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set4": ['Total number of Nodes', 'mean_distance', 'median_distance', 'std_distance', 'min_distance',
                     'max_distance', 'sum_distance', 'density', 'percentile_25', 'percentile_50', 'percentile_75',
                     'Total Time Window', 'Average Time Window', 'Max Time Window', 'Min Time Window',
                     'Standard Deviation of Time Window', 'Mean Latitude', 'Mean Longitude',
                     'Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set5": ['Total number of Nodes', 'MinP', 'MaxP', 'VarP', 'SumMinP', 'SumMaxP', 'MinM', 'MaxM', 'SumM',
                     'VarM', 'VarX×VarY', 'Total Time Window', 'Average Time Window', 'Max Time Window',
                     'Min Time Window', 'Standard Deviation of Time Window', 'Mean Latitude', 'Mean Longitude',
                     'Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set6": ['Total number of Nodes', 'MinP', 'MaxP', 'MinM', 'MaxM', 'SumM', 'VarM', 'VarX×VarY',
                     'mean_distance', 'std_distance', 'min_distance', 'max_distance', 'sum_distance', 'density',
                     'percentile_25', 'percentile_50', 'percentile_75', 'Total Time Window', 'Average Time Window',
                     'Max Time Window', 'Min Time Window', 'Standard Deviation of Time Window', 'Mean Latitude',
                     'Mean Longitude', 'Standard Deviation of Latitude', 'Standard Deviation of Longitude',
                     'Min Latitude', 'Max Latitude', 'Min Longitude', 'Max Longitude', 'AvgDepotDist', 'MaxDepotDist',
                     'MinDepotDist', 'StdDevDepotDist'],
            "set7": ['Total number of Nodes', 'mean_distance', 'median_distance', 'std_distance', 'min_distance',
                     'max_distance', 'sum_distance', 'Total Time Window', 'Average Time Window', 'Max Time Window',
                     'Min Time Window',
                     'Standard Deviation of Time Window', 'Mean Latitude', 'Mean Longitude',
                     'Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude', 'Average Distance from Center',
                     'Max Distance from Center', 'Min Distance from Center',
                     'Standard Deviation of Distance from Center', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set8": ['Total number of Nodes', 'VarP', 'SumMinP', 'SumMaxP', 'MinM', 'MaxM', 'SumM',
                     'VarM', 'std_distance', 'min_distance',
                     'max_distance', 'sum_distance', 'density', 'percentile_25', 'percentile_50', 'percentile_75',
                     'Total Time Window', 'Average Time Window', 'Max Time Window', 'Min Time Window',
                     'Standard Deviation of Time Window','Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude','AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set9": ['Total number of Nodes', 'MinP', 'MaxP', 'VarP', 'SumMinP', 'SumMaxP', 'MinM', 'MaxM', 'SumM',
                     'VarM', 'VarX×VarY', 'mean_distance', 'median_distance', 'std_distance', 'min_distance',
                     'max_distance', 'sum_distance', 'density', 'percentile_25', 'percentile_50', 'percentile_75',
                     'Total Time Window', 'Average Time Window', 'Max Time Window', 'Min Time Window',
                     'Standard Deviation of Time Window', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set10": ['Total number of Nodes','min_distance', 'max_distance', 'Average Time Window', 'Max Time Window', 'Min Time Window',
                     'Standard Deviation of Time Window', 'Standard Deviation of Latitude', 'Standard Deviation of Longitude', 'Min Latitude',
                     'Max Latitude', 'Min Longitude', 'Max Longitude','AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                     'StdDevDepotDist'],
            "set11": ['Total number of Nodes', 'min_distance', 'max_distance', 'Average Time Window', 'Max Time Window',
                      'Min Time Window','Standard Deviation of Time Window', 'Standard Deviation of Latitude',
                      'Standard Deviation of Longitude', 'AvgDepotDist', 'MaxDepotDist', 'MinDepotDist',
                      'StdDevDepotDist'],


        }

    def get_feature_set(self, set_name):
        return self.feature_sets.get(set_name, [])


def load_data():
    train_data = pd.read_csv('../Data/Results/Training_input_features_2.csv')
    test_data = pd.read_csv('../Data/Results/Testing_input_features_2.csv')

    train_data.replace(-1, pd.NA, inplace=True)
    test_data.replace(-1, pd.NA, inplace=True)

    # Fill NaN values with the median of each column
    train_data.fillna(train_data.median(), inplace=True)
    test_data.fillna(test_data.median(), inplace=True)

    return train_data, test_data
