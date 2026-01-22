import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.regression.data_handler import DataLoader
from src.regression.data_splitter import DataSplitter
from src.regression.regression_visualizer import RegressionVisualizer


class PolynomialRegressionModel:
    def __init__(self, degree: int = 2):
        self._degree = degree
        self._poly_features = PolynomialFeatures(degree=self._degree)
        self._model = LinearRegression()

    def train(self, X: DataFrame, y: Series) -> None:
        X_poly = self._poly_features.fit_transform(X)
        self._model.fit(X_poly, y)

    def predict(self, X: DataFrame) -> np.ndarray:
        X_poly = self._poly_features.transform(X)
        return self._model.predict(X_poly)

    def evaluate(self, y_true: Series, y_pred: np.ndarray) -> dict:
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }


data = DataLoader('../data/processed/insurance_encoded.csv').load_csv()
X_train, X_test, y_train, y_test = DataSplitter(target_column='charges').split(data)

poly_model = PolynomialRegressionModel(degree=2)
poly_model.train(X_train, y_train)

y_pred_poly = poly_model.predict(X_test)

sample_index = 0
sample_input = X_test.iloc[[sample_index]]  # DataFrame required by predict
actual_value = y_test.iloc[sample_index]
predicted_value = poly_model.predict(sample_input)[0]

print(f"Actual value: {actual_value:.2f}")
print(f"Predicted value: {predicted_value:.2f}")
print(f"Error (absolute difference): {abs(actual_value - predicted_value):.2f}")

metrics = poly_model.evaluate(y_test, y_pred_poly)
print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
print(f"R^2 Score: {metrics['r2']:.4f}")

visualizer = RegressionVisualizer()
visualizer.plot_prediction_vs_actual(y_test, y_pred_poly)
visualizer.plot_residuals(y_test, y_pred_poly)
