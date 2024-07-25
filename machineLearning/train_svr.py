# train_svr.py
import os
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from models_evaluate_plot import evaluate_model#, plot_results
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#def train_svr(X_train, y_train, X_valid, y_valid, feature_set_name,model_dir):
def train_svr(X_train, y_train, X_valid, y_valid, X_test, feature_set_name):
    param_distributions = {
        'C': uniform(0.1, 10),
        'epsilon': uniform(0.1, 1)
    }

    svr = SVR()
    random_search = RandomizedSearchCV(estimator=svr, param_distributions=param_distributions, n_iter=100, cv=5,
                                       n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_valid_pred = best_model.predict(X_valid)
    valid_metrics = evaluate_model(y_valid, y_valid_pred, feature_set_name, "SVR")

    y_test_pred = best_model.predict(X_test)
    test_predictions = {"Predicted": y_test_pred.tolist()}  # Store test predictions

    return best_model, valid_metrics, test_predictions
"""
     #save model

    model_path = os.path.join(model_dir, f"svr_model_{feature_set_name}.joblib")
    joblib.dump(svr, model_path)
    print(f"SVR model saved to {model_path}")
    """


