import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class DecisionTreeBinaryModel:
    def __init__(
        self,
        max_depth: int = 3,
        random_state: int = 42,
        class_weight: str = "balanced"
    ):
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight
        )
        self.class_names = ["Healthy", "Diseased"]

    def load_data(self, file_path: str) -> DataFrame:
        return pd.read_csv(file_path)

    def prepare_features_target(self, df: DataFrame, target: str):
        df = df.copy()
        df[target] = (df[target] > 0).astype(int)

        X = df.drop(columns=[target])
        y = df[target]

        print("Binary classification: Healthy vs Diseased")
        print(y.value_counts())

        return X, y

    def split_data(self, X: DataFrame, y: pd.Series):
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Decision Tree model trained.")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)

        print("-" * 30)
        print(f"Decision Tree Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=self.class_names,
                zero_division=0
            )
        )

        return accuracy

    def plot_tree_model(self, feature_names):
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f"Binary Decision Tree (max_depth={self.max_depth})")
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Decision Tree – Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_names, top_n: int = 10):
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="importance",
            y="feature",
            data=importances.head(top_n),
            palette="viridis",
            legend=False
        )
        plt.title(f"Top {top_n} Most Important Features")
        plt.tight_layout()
        plt.show()

        return importances


dt = DecisionTreeBinaryModel(max_depth=3)

df = dt.load_data("../data/processed/heart_disease_uci_encoded.csv")

X, y = dt.prepare_features_target(df, target="num")

X_train, X_test, y_train, y_test = dt.split_data(X, y)

dt.train(X_train, y_train)

dt.plot_tree_model(feature_names=X.columns)

y_pred = dt.predict(X_test)

dt.evaluate(y_test, y_pred)

dt.plot_confusion_matrix(y_test, y_pred)

importances = dt.plot_feature_importance(
    feature_names=X.columns,
    top_n=10
)


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

prediction = dt.predict(new_patient)[0]
proba = dt.model.predict_proba(new_patient)[0]

print("Model decision:")
print("Diseased" if prediction == 1 else "Healthy")

print(f"Healthy probability: {proba[0]:.2%}")
print(f"Diseased probability: {proba[1]:.2%}")
