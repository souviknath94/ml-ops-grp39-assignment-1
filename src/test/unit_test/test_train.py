import unittest
from src.train import train_model
from src.app import predict
import joblib
import os


class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        """
        Test the train_model function to ensure it returns the best parameters and accuracy.
        """
        best_params, accuracy = train_model()
        self.assertIsInstance(best_params, dict, "Best parameters should be a dictionary.")
        self.assertTrue(0 <= accuracy <= 1, "Accuracy should be a float between 0 and 1.")
        self.assertTrue(os.path.exists('model.pkl'), "Model file should exist after training.")

    def test_predict(self):
        """
        Test the predict function with valid input features.
        """
        # Train the model to ensure model.pkl exists
        train_model()

        # Predict with valid input
        features = [13.2, 2.77, 2.51, 24.5, 86.0, 1.45, 1.25, 0.5, 1.7, 6.5, 1.05, 3.33, 820.0]
        prediction = predict(features)
        self.assertIsInstance(prediction, int, "Prediction should be an integer (class label).")

    def test_predict_invalid_input(self):
        """
        Test the predict function with invalid input to ensure it raises an exception.
        """
        with self.assertRaises(ValueError):
            predict(["invalid", "data"])


if __name__ == '__main__':
    unittest.main()
