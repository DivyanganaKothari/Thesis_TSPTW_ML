# train_rf.py
import os
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from models_evaluate_plot import evaluate_model

def train_rf(X_train, y_train, X_valid, y_valid, X_test, feature_set_name):

    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=100, cv=5,
                                       n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_valid_pred = best_model.predict(X_valid)
    y_test_pred = best_model.predict(X_test)

    valid_metrics = evaluate_model(y_valid, y_valid_pred, feature_set_name, "Random Forest")
    test_predictions = {"Predicted": y_test_pred.tolist()}

    return best_model, valid_metrics, test_predictions
