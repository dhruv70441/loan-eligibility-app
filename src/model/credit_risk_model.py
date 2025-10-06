from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

class CreditRiskModel:
    def __init__(self):

        self.models = {
            'LogisticsRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
        }

        self.best_model = None
        self.best_model_name = None


    def train_and_select(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            }

            results[name] = metrics

            self.best_model_name = max(results, key=lambda m: results[m]["f1_score"])
            self.best_model = self.models[self.best_model_name]

            return results, self.best_model_name
        

    def save_model(self, filepath="models/credit_risk_model.pkl"):
        if self.best_model:
            joblib.dump(self.best_model, filepath)
            print(f"Best Model ({self.best_model_name}) saved at {filepath}")
        else:
            print("No model trained yet. Train and select a model first.")

    def load_model(self, filepath="models/credit_risk_model.pkl"):
        self.best_model = joblib.load(filepath)
        print(f"🔄 Model loaded from {filepath}")
        return self.best_model