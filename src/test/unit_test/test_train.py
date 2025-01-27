import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.train import train_model
from src.app import model_predict
import os


class TestTrainModel(unittest.TestCase):

    @patch('src.train.GridSearchCV')
    @patch('src.train.mlflow')
    @patch('src.train.joblib')
    @patch('src.train.train_test_split')
    def test_train_model(self, mock_train_test_split, mock_joblib, mock_mlflow, mock_grid_search):
        """
        Test the train_model function to ensure it returns the best parameters and accuracy.
        """
        # Mock the train-test split
        mock_train_test_split.return_value = (
            "X_train",
            "X_test",
            [1, 0, 1, 0],  # y_train
            [1, 0, 1, 0]   # y_test
        )

        # Mock the GridSearchCV fit and results
        mock_grid_search_instance = MagicMock()
        mock_grid_search_instance.best_params_ = {"param1": "value1"}
        mock_grid_search_instance.best_estimator_ = MagicMock()
        mock_grid_search_instance.best_estimator_.predict.return_value = [1, 0, 1, 1]  # y_pred
        mock_grid_search.return_value = mock_grid_search_instance

        # Mock MLflow methods
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None

        # Mock joblib
        mock_joblib.dump.return_value = None

        # Call the function under test
        from src.train import train_model
        result = train_model("MockModel", MagicMock(), {"param1": [1, 2, 3]})

        # Ensure train_model returns a valid tuple
        self.assertIsNotNone(result, "train_model should not return None")
        best_params, accuracy = result  # Unpack the result

        # Assertions
        self.assertEqual(best_params, {"param1": "value1"})
        self.assertIsInstance(accuracy, float, "Accuracy should be a float.")
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be between 0 and 1.")
        mock_joblib.dump.assert_called_once_with(mock_grid_search_instance.best_estimator_, "models/model.pkl")
        mock_mlflow.log_param.assert_called_with("model_name", "MockModel")
        mock_mlflow.log_params.assert_called_with({"param1": "value1"})
        mock_mlflow.log_metric.assert_called_once_with("accuracy", 0.75)

    def test_predict(self):
        """
        Test the predict function with valid input features.
        """
        # Predict with valid input
        features = [13.2, 2.77, 2.51, 24.5, 86.0, 1.45, 1.25, 0.5, 1.7, 6.5, 1.05, 3.33, 820.0]
        prediction = model_predict(np.array(features).reshape(1, -1))
        self.assertIsInstance(prediction, np.ndarray, "Prediction should be an array of ints (class label).")

    def test_predict_invalid_input(self):
        """
        Test the predict function with invalid input to ensure it raises an exception.
        """
        with self.assertRaises(ValueError):
            model_predict(["invalid", "data"])


if __name__ == '__main__':
    unittest.main()
