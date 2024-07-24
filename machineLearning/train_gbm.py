# train_gbm.py
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint
from models_evaluate_plot import evaluate_model

def train_gbm(X_train, y_train, X_valid, y_valid, feature_set_name):
    param_distributions = {
        'n_estimators': randint(100, 500),
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    gbm = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=gbm, param_distributions=param_distributions, n_iter=100, cv=5,
                                       n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    """
    # Save model
    model_path = os.path.join(model_dir, f"best_gbm_model_{feature_set_name}.joblib")
    joblib.dump(best_model, model_path)
    print(f"GBM model saved to {model_path}")
    """

    y_valid_pred = best_model.predict(X_valid)
    return best_model, evaluate_model(y_valid, y_valid_pred, feature_set_name, "GBM")
