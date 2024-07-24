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
"""
def train_svr(X_train, y_train, X_valid, y_valid, feature_set_name):
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

     #save model

    model_path = os.path.join(model_dir, f"svr_model_{feature_set_name}.joblib")
    joblib.dump(svr, model_path)
    print(f"SVR model saved to {model_path}")

    return best_model, evaluate_model(y_valid, y_valid_pred, feature_set_name, "SVR")
"""
def train_svr(X_train, y_train, X_valid, y_valid, feature_set_name):
    # Use a pipeline to ensure proper scaling
    svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))

    svr_pipeline.fit(X_train, y_train)
    y_valid_pred = svr_pipeline.predict(X_valid)

    return svr_pipeline, evaluate_model(y_valid, y_valid_pred, feature_set_name, "SVR")
