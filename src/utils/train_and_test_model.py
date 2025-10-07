import warnings
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.data_cleaning import DataCleaner
from src.data.feature_engineering import FeatureProcessor
from src.model.credit_risk_model import CreditRiskModel
from src.pipeline.prediction_pipeline import PredictionPipeline

warnings.filterwarnings('ignore')

# ==============================
# Step 0: Load dataset
# ==============================
df = pd.read_csv(r'data/loan-train.csv')
master_df = df.copy()
print("Dataset loaded. Sample data:")
print(master_df.head())

# ==============================
# Step 1: Data Cleaning
# ==============================
print("\n--- Data Cleaning ---")
print('Null values before cleaning:', df.isnull().sum().sum())
print('Duplicate rows before cleaning:', df.duplicated().sum())

cleaner = DataCleaner(df)
cleaned_df = cleaner.handle_missing_values().remove_duplicates().get_processed_data()

print('Null values after cleaning:', cleaned_df.isnull().sum().sum())
print('Duplicate rows after cleaning:', cleaned_df.duplicated().sum())

# ==============================
# Step 2: Feature Scaling & Encoding
# ==============================
print("\n--- Feature Scaling & Encoding ---")
processor = FeatureProcessor(cleaned_df)
processed_df = processor.scale_features().encode_categorical().get_features()

print("Sample of processed data:")
print(processed_df.sample(5))

# ==============================
# Step 3: Model Training & Selection
# ==============================
print("\n--- Model Training & Selection ---")
X = processed_df.drop(columns=['Loan_Status'])
y = processed_df['Loan_Status']

model = CreditRiskModel()
results, best_model_name = model.train_and_select(X, y)

print(f"\nBest model selected: {best_model_name}")
print("Evaluation metrics:")
print(results[best_model_name])

# ==============================
# Step 4: Save Trained Model
# ==============================
model.save_model()

# ==============================
# Step 5: Manual Prediction (Using Pipeline)
# ==============================
print("\n--- Manual Prediction ---")
pp = PredictionPipeline()

# Predict on the same dataset for demonstration
preds, probs = pp.make_prediction(X)

# Ensure target is encoded as 0/1
y_true = y.replace({'Y': 1, 'N': 0})

# ==============================
# Step 6: Evaluation Metrics
# ==============================
accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)
roc_auc = roc_auc_score(y_true, probs)

print("\nPrediction Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
