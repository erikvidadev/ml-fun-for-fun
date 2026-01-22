import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class KNNBinaryModel:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.class_names = ["Healthy", "Diseased"]

    def load_csv_data(self, file_path: str) -> DataFrame:
        return pd.read_csv(file_path)

    def split_train_test_data(self, X: DataFrame, y: pd.Series):
        return train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    def scale_data(self, X_train: DataFrame, X_test: DataFrame):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train(self, X_train_scaled, y_train):
        self.knn_model.fit(X_train_scaled, y_train)

    def predict(self, X_test_scaled):
        return self.knn_model.predict(X_test_scaled)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"KNN Accuracy: {accuracy:.2%}\n")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        return accuracy

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("KNN Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def find_best_k(self, X_train_scaled, y_train, X_test_scaled, y_test, k_max: int = 40):
        error_rate = []
        for k in range(1, k_max):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            pred_k = knn.predict(X_test_scaled)
            error_rate.append(np.mean(pred_k != y_test))

        best_k = error_rate.index(min(error_rate)) + 1
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, k_max), error_rate, linestyle='dashed', marker='o')
        plt.xlabel("Number of Neighbors (K)")
        plt.ylabel("Error Rate")
        plt.title("Error Rate vs K Value")
        plt.tight_layout()
        plt.show()

        print(f"Optimal K: {best_k}")
        return best_k, error_rate


knn = KNNBinaryModel(n_neighbors=5)

df = knn.load_csv_data("../data/processed/heart_disease_uci_encoded.csv")

X = df.drop("num", axis=1)
y = (df["num"] > 0).astype(int)

X_train, X_test, y_train, y_test = knn.split_train_test_data(X, y)
X_train_scaled, X_test_scaled = knn.scale_data(X_train, X_test)

knn.train(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

knn.evaluate(y_test, y_pred)
knn.plot_confusion_matrix(y_test, y_pred)

best_k, error_rate = knn.find_best_k(X_train_scaled, y_train, X_test_scaled, y_test, k_max=40)


new_patient = pd.DataFrame([{
    "age": 58,
    "sex_Female": 0,
    "sex_Male": 1,
    "trestbps": 145,
    "chol": 250,
    "thalch": 140,
    "fbs": 0,
    "exang": 0,
    "oldpeak": 1.8,
    "cp_typical angina": 0,
    "cp_atypical angina": 1,
    "cp_non-anginal": 0,
    "cp_asymptomatic": 0,
    "restecg_normal": 1,
    "restecg_st-t abnormality": 0,
    "restecg_lv hypertrophy": 0,
    "slope_upsloping": 0,
    "slope_flat": 0,
    "slope_downsloping": 1,
    "missing_ca": 0,
    "ca": 0,
    "missing_thal": 0,
    "thal_normal": 0,
    "thal_fixed defect": 0,
    "thal_reversable defect": 1
}])

prediction = knn.predict(knn.scaler.transform(new_patient))[0]
proba = knn.knn_model.predict_proba(knn.scaler.transform(new_patient))[0]

print("Model decision:")
print("Diseased" if prediction == 1 else "Healthy")
print(f"Healthy probability: {proba[0]:.2%}")
print(f"Diseased probability: {proba[1]:.2%}")
