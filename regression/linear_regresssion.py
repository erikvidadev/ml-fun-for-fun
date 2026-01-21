from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from regression.data_handler import DataLoader
from regression.data_splitter import DataSplitter
from regression.regression_visualizer import RegressionVisualizer


class LinearRegressionModel:
    def __init__(self):
        self._model = LinearRegression()

    def train(self, X: DataFrame, y: Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def evaluate(self, y_true: Series, y_pred: np.ndarray) -> dict:
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }



data = DataLoader("../data/processed/insurance_encoded.csv").load_csv()
X_train, X_test, y_train, y_test = DataSplitter(target_column="charges").split(data)

model = LinearRegressionModel()
model.train(X_train, y_train)

y_pred_test = model.predict(X_test)
metrics = model.evaluate(y_test, y_pred_test)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
print(f"R^2 Score: {metrics['r2']:.4f}")

new_profile = {col: [0] for col in X_train.columns}
new_profile.update({
    "age": [40],
    "bmi": [30.5],
    "smoker": [1],
})
new_input_df = DataFrame(new_profile)
predicted_charge = model.predict(new_input_df)

print("-" * 30)
print(f"Predicted insurance charge for the custom profile: {predicted_charge[0]:.2f}")

visualizer = RegressionVisualizer()
visualizer.plot_prediction_vs_actual(y_test, y_pred_test)
visualizer.plot_residuals(y_test, y_pred_test)
