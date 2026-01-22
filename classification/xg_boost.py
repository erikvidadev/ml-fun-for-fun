import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance


class XGBoostBinaryModel:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss"
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

    def split_data(self, X: DataFrame, y: pd.Series, test_size: float = 0.2):
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("XGBoost model trained.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost Accuracy: {accuracy:.2%}\n")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        return accuracy

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='mako',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("XGBoost Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n: int = 10):
        plt.figure(figsize=(10, 8))
        plot_importance(self.model, importance_type="gain", max_num_features=top_n)
        plt.title(f"XGBoost: Top {top_n} Most Important Features")
        plt.tight_layout()
        plt.show()


xgb_model = XGBoostBinaryModel()

df = xgb_model.load_data("../data/processed/heart_disease_uci_encoded.csv")

X, y = xgb_model.prepare_features_target(df, target="num")

X_train, X_test, y_train, y_test = xgb_model.split_data(X, y)

xgb_model.train(X_train, y_train)

y_pred = xgb_model.predict(X_test)

xgb_model.evaluate(y_test, y_pred)
xgb_model.plot_confusion_matrix(y_test, y_pred)
xgb_model.plot_feature_importance(top_n=10)


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

prediction = xgb_model.predict(new_patient)[0]
proba = xgb_model.predict_proba(new_patient)[0]

print("Model decision:")
print("Diseased" if prediction == 1 else "Healthy")
print(f"Healthy probability: {proba[0]:.2%}")
print(f"Diseased probability: {proba[1]:.2%}")
