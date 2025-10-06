import pandas as pd
from src.data.data_cleaning import DataCleaner
from src.data.feature_engineering import FeatureProcessor
from src.model.credit_risk_model import CreditRiskModel

class PredictionPipeline:
    def __init__(self, model_path="models/credit_risk_model.pkl"):
        self.crm = CreditRiskModel()
        self.model = self.crm.load_model(model_path)

    def make_prediction(self, df: pd.DataFrame):
        """
        df: pandas DataFrame with same columns used in training
        Returns: predictions (0/1) and probabilities
        """
        # Preprocess features
        cleaner = DataCleaner(df)
        X_cleaned = (
            cleaner.handle_missing_values()
            .remove_duplicates()
            .get_processed_data()
        )

        processor = FeatureProcessor(X_cleaned)
        X_processed = (
            processor.scale_features()
            .encode_categorical()
            .get_features()
        )

        # Predict
        preds = self.model.predict(X_processed)
        probs = self.model.predict_proba(X_processed)[:,1]

        return preds, probs
