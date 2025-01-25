import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_model(model_name, model, param_grid):
    """
    Train a model using the wine dataset and log the process with MLflow.
    Args:
        model_name (str): Name of the model (e.g., "RandomForest", "LogisticRegression", "SVM").
        model (object): Model instance to train.
        param_grid (dict): Hyperparameter grid for the model.
    """
    # Save the dataset as a CSV file
    dataset_path = "data/wine_dataset.csv"
    df = pd.read_csv(dataset_path)

    # Split dataset
    X, y = df.drop(columns=['target']), df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow tracking
    remote_server_uri="http://127.0.0.1:5000" 
    mlflow.set_tracking_uri(remote_server_uri) 
    mlflow.set_experiment("Wine_Classification")
    with mlflow.start_run():
        # Perform Grid Search
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=0)
        grid_search.fit(X_train, y_train)

        # Best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log model name
        mlflow.log_param("model_name", model_name)

        # Log best parameters
        mlflow.log_params(best_params)

        # Log test accuracy
        mlflow.log_metric("accuracy", accuracy)

        # Log the trained model
        mlflow.sklearn.log_model(best_model, f"{model_name}_model")

        # Save the model locally
        joblib.dump(best_model, f"{model_name}_model.pkl")

        print(f"{model_name} trained. Best Parameters: {best_params}, Test Accuracy: {accuracy}")


if __name__ == "__main__":
    # Define models and their parameter grids
    models = [
        {
            "name": "model",
            "model": SVC(random_state=42),
            "param_grid": {
                'C': [0.1, 1.0, 10],
                'kernel': ['linear', 'rbf']
            }
        }
    ]

    # Train and log each model
    for model_info in models:
        train_model(model_info["name"], model_info["model"], model_info["param_grid"])
