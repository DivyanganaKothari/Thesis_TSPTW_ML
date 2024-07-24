# main.py
import os
import pandas as pd
from models_feature_manager import FeatureManager, load_data
from sklearn.model_selection import train_test_split
from train_rf import train_rf
from train_svr import train_svr
from train_gnn import train_gnn
from train_nn import train_nn
from train_gbm import train_gbm
from train_decision_tree import train_decision_tree


#from models_evaluate_plot import plot_results


def main():
    train_data, test_data = load_data()

    # model_dir = '../Data/Models'

    # graphs_dir = '../Data/Graphs'
    results_dir = '../Data/Results'

    # if not os.path.exists(model_dir):
    #    os.makedirs(model_dir)
    #if not os.path.exists(graphs_dir):
     #   os.makedirs(graphs_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    feature_manager = FeatureManager()
    target = 'tour_length'
    results = []

    for feature_set_name in feature_manager.feature_sets:
        feature_set = feature_manager.get_feature_set(feature_set_name)
        X = train_data[feature_set]
        y = train_data[target]

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        #_, rf_metrics = train_rf(X_train, y_train, X_valid, y_valid, feature_set_name, model_dir)
        _, rf_metrics = train_rf(X_train, y_train, X_valid, y_valid, feature_set_name)

        results.append(rf_metrics)
        #plot_results(y_valid, rf_metrics["y_valid_pred"], feature_set_name, "Random Forest", graphs_dir)

        # Train SVR
        # _, svr_metrics = train_svr(X_train, y_train, X_valid, y_valid, feature_set_name, model_dir)
        _, svr_metrics = train_svr(X_train, y_train, X_valid, y_valid, feature_set_name)
        results.append(svr_metrics)
        #plot_results(y_valid, svr_metrics["y_valid_pred"], feature_set_name, "SVR", graphs_dir)

        # Train GNN
        #_, gnn_metrics = train_gnn(X_train, y_train, X_valid, y_valid, feature_set_name, model_dir)
        _, gnn_metrics = train_gnn(X_train, y_train, X_valid, y_valid, feature_set_name)
        results.append(gnn_metrics)
        #plot_results(y_valid, gnn_metrics["y_valid_pred"], feature_set_name, "GNN", graphs_dir)

        # Train Neural Network
        _, nn_metrics = train_nn(X_train, y_train, X_valid, y_valid, feature_set_name)
        results.append(nn_metrics)

        # Train GBM
        _, gbm_metrics = train_gbm(X_train, y_train, X_valid, y_valid, feature_set_name)
        results.append(gbm_metrics)

        # Train Decision Tree
        _, dt_metrics = train_decision_tree(X_train, y_train, X_valid, y_valid, feature_set_name)
        results.append(dt_metrics)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(results_dir, 'model_comparison_results_2.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Comparison results saved to {results_path}")


if __name__ == "__main__":
    main()
