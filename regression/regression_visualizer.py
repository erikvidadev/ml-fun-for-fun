import numpy as np
import seaborn as sns
from pandas import Series
from matplotlib import pyplot as plt


class RegressionVisualizer:
    @staticmethod
    def plot_prediction_vs_actual(y_true: Series, y_pred: np.ndarray):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.title("Actual vs Predicted Values (R²)")
        plt.show()

    @staticmethod
    def plot_residuals(y_true: Series, y_pred: np.ndarray):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.axvline(0, color='r', linestyle='--')
        plt.title("Residuals Distribution")
        plt.xlabel("Error")
        plt.show()
