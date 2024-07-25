import os
import pandas as pd
from models_feature_manager import FeatureManager, load_data
from sklearn.model_selection import train_test_split
from train_rf import train_rf
from train_svr import train_svr
from train_gnn import train_gnn
from train_nn import train_nn
from train_gbm import train_gbm

def main():
    train_data, test_data = load_data()

    results_dir = '../Data/Results'
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
        X_test = test_data[feature_set]

        # Train Random Forest
        best_rf_model, rf_valid_metrics, rf_test_predictions = train_rf(X_train, y_train, X_valid, y_valid, X_test, feature_set_name)
        results.append(rf_valid_metrics)
        #save_predictions(rf_test_predictions, feature_set_name, "Random Forest", results_dir)

        # Train SVR
        best_svr_model, svr_valid_metrics, svr_test_predictions = train_svr(X_train, y_train, X_valid, y_valid, X_test, feature_set_name)
        results.append(svr_valid_metrics)
        #save_predictions(svr_test_predictions, feature_set_name, "SVR", results_dir)

        # Train GNN
        best_gnn_model, gnn_valid_metrics, gnn_test_predictions = train_gnn(X_train, y_train, X_valid, y_valid, X_test, feature_set_name)
        results.append(gnn_valid_metrics)
        #save_predictions(gnn_test_predictions, feature_set_name, "GNN", results_dir)

        # Train Neural Network
        best_nn_model, nn_valid_metrics, nn_test_predictions = train_nn(X_train, y_train, X_valid, y_valid, X_test, feature_set_name)
        results.append(nn_valid_metrics)
        #save_predictions(nn_test_predictions, feature_set_name, "Neural Network", results_dir)

        # Train GBM
        best_gbm_model, gbm_valid_metrics, gbm_test_predictions = train_gbm(X_train, y_train, X_valid, y_valid, X_test, feature_set_name)
        results.append(gbm_valid_metrics)
        #save_predictions(gbm_test_predictions, feature_set_name, "GBM", results_dir)


    results_df = pd.DataFrame(results)
    results_path = os.path.join(results_dir, 'model_comparison_results_set2.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Comparison results saved to {results_path}")

def save_predictions(predictions, feature_set_name, model_name, results_dir):
    if predictions is not None:
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
        predictions_path = os.path.join(results_dir, f'{model_name}_test_predictions_{feature_set_name}.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"{model_name} test predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()
