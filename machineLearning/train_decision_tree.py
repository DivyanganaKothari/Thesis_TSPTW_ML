# train_decision_tree.py
import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from models_evaluate_plot import evaluate_model

def train_decision_tree(X_train, y_train, X_valid, y_valid, feature_set_name):
    param_distributions = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    dt = DecisionTreeRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=dt, param_distributions=param_distributions, n_iter=100, cv=5,
                                       n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    """
    # Save model
    model_path = os.path.join(model_dir, f"best_dt_model_{feature_set_name}.joblib")
    joblib.dump(best_model, model_path)
    print(f"Decision Tree model saved to {model_path}")
    """

    y_valid_pred = best_model.predict(X_valid)
    return best_model, evaluate_model(y_valid, y_valid_pred, feature_set_name, "Decision Tree")
