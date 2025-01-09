import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    """
    Train a Random Forest model using the wine dataset and save it to a file.
    Returns the best parameters and test accuracy.
    """
    # Load dataset
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model and parameter grid
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model
    joblib.dump(best_model, 'model.pkl')

    return best_params, accuracy


def predict(features):
    """
    Predict the wine class for given features using the trained model.
    """
    try:
        model = joblib.load('model.pkl')
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")