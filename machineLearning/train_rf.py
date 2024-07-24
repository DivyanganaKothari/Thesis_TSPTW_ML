# train_rf.py
import os
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from models_evaluate_plot import evaluate_model#, plot_results



#def train_rf(X_train, y_train, X_valid, y_valid, feature_set_name, model_dir):
def train_rf(X_train, y_train, X_valid, y_valid, feature_set_name):
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=100, cv=5,
                                       n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    """
     #save model
    model_path = os.path.join(model_dir, f"best_rf_model_{feature_set_name}.joblib")
    joblib.dump(best_model, model_path)
    print(f"Random Forest model saved to {model_path}")
    """

    y_valid_pred = best_model.predict(X_valid)
    return best_model, evaluate_model(y_valid, y_valid_pred, feature_set_name, "Random Forest")
