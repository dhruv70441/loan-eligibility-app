from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

class CreditRiskModel:
    def __init__(self):
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
        }

        self.param_grids = {
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.results = {}

    def tune_and_train_model(self, name, model, X_train, y_train):
        """Tune a single model using GridSearchCV"""
        print(f"\n Tuning hyperparameters for {name}...")
        grid = GridSearchCV(
            estimator=model,
            param_grid=self.param_grids[name],
            scoring='f1',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        print(f"Best {name} Params: {grid.best_params_}")
        return grid.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        """Compute key evaluation metrics"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
        }

    def train_and_select(self, X, y, test_size=0.2, random_state=42):
        """Train all models, tune them, and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        for name, model in self.models.items():
            tuned_model = self.tune_and_train_model(name, model, X_train, y_train)
            metrics = self.evaluate_model(tuned_model, X_test, y_test)
            self.results[name] = metrics

        # Select best model by F1-score
        self.best_model_name = max(self.results, key=lambda m: self.results[m]["f1_score"])
        self.best_model = self.models[self.best_model_name]
        self.best_params = self.param_grids[self.best_model_name]

        print(f"\nBest Model: {self.best_model_name}")
        print("Metrics:", self.results[self.best_model_name])
        return self.results, self.best_model_name

    def save_model(self, filepath="models/credit_risk_model.pkl"):
        if self.best_model:
            joblib.dump(self.best_model, filepath)
            print(f"Best Model ({self.best_model_name}) saved at {filepath}")
        else:
            print("No model trained yet. Train and select a model first.")

    def load_model(self, filepath="models/credit_risk_model.pkl"):
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.best_model
