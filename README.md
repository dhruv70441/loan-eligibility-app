# 🏦 Loan Approval Prediction System

A comprehensive machine learning system for predicting loan approval decisions using various classification algorithms. The system includes data preprocessing, feature engineering, model training with hyperparameter tuning, and a user-friendly Streamlit web interface.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)

## ✨ Features

- **Multiple ML Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- **Automated Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Data Preprocessing**: Missing value handling, categorical encoding, and feature scaling
- **Interactive Web Interface**: Streamlit-based UI for single predictions and bulk processing
- **Model Persistence**: Save and load trained models using joblib
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics

## 🏗️ Project Structure

```
loan approval prediction/
├── data/
│   ├── loan-train.csv          # Training dataset
│   └── loan-test.csv           # Test dataset
├── models/
│   └── credit_risk_model.pkl   # Trained model file
├── src/
│   ├── app/
│   │   └── app.py              # Streamlit web application
│   ├── data/
│   │   ├── data_cleaning.py    # Data preprocessing utilities
│   │   └── feature_engineering.py  # Feature scaling and encoding
│   ├── model/
│   │   └── credit_risk_model.py    # ML model training and selection
│   ├── pipeline/
│   │   └── prediction_pipeline.py  # Prediction pipeline
│   └── utils/
│       └── train_and_test_model.py # Training and evaluation script
├── requirements.txt            # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhruv70441/loan-eligibility-app.git
   cd "loan approval prediction"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, sklearn, xgboost, streamlit; print('All packages installed successfully!')"
   ```

## 📖 Usage

### 1. Training the Model

Run the training script to train and evaluate multiple models:

```bash
python src/utils/train_and_test_model.py
```

This script will:
- Load and preprocess the training data
- Train multiple models with hyperparameter tuning
- Select the best performing model
- Save the trained model to `models/credit_risk_model.pkl`
- Display evaluation metrics

### 2. Web Application

Launch the Streamlit web interface:

```bash
streamlit run src/app/app.py
```

The application will open in your browser at `http://localhost:8501`

#### Features of the Web App:

**Single Entry Mode:**
- Input form for individual loan applications
- Real-time prediction with approval probability
- User-friendly interface with dropdown menus

**Bulk CSV Upload Mode:**
- Upload CSV files with multiple loan applications
- Batch processing and prediction
- Download results as CSV file

### 3. Programmatic Usage

```python
from src.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd

# Initialize prediction pipeline
pp = PredictionPipeline("models/credit_risk_model.pkl")

# Prepare data (same format as training data)
data = {
    'Gender': ['Male'],
    'Married': ['Yes'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [5000],
    'CoapplicantIncome': [2000],
    'LoanAmount': [100000],
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    'Property_Area': ['Urban']
}

df = pd.DataFrame(data)

# Make prediction
predictions, probabilities = pp.make_prediction(df)
print(f"Prediction: {'Approved' if predictions[0]==1 else 'Rejected'}")
print(f"Probability: {probabilities[0]*100:.2f}%")
```

## 📊 Model Performance

The system automatically trains and evaluates multiple models:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| Gradient Boosting | - | - | - | - | - |
| XGBoost | - | - | - | - | - |

*Note: Actual performance metrics will be displayed after running the training script*

## 🔧 API Documentation

### DataCleaner Class

Handles data preprocessing operations:

```python
cleaner = DataCleaner(df)
cleaned_df = cleaner.handle_missing_values().remove_duplicates().get_processed_data()
```

**Methods:**
- `handle_missing_values()`: Fills missing values using median (numeric) and mode (categorical)
- `remove_duplicates()`: Removes duplicate rows
- `get_processed_data()`: Returns cleaned DataFrame

### FeatureProcessor Class

Handles feature engineering:

```python
processor = FeatureProcessor(df)
processed_df = processor.scale_features().encode_categorical().get_features()
```

**Methods:**
- `encode_categorical()`: Label encodes categorical variables
- `scale_features()`: Standardizes numerical features
- `get_features()`: Returns processed DataFrame

### CreditRiskModel Class

Manages model training and selection:

```python
model = CreditRiskModel()
results, best_model = model.train_and_select(X, y)
```

**Methods:**
- `train_and_select(X, y)`: Trains all models and selects the best one
- `save_model(filepath)`: Saves the best model to file
- `load_model(filepath)`: Loads a saved model

### PredictionPipeline Class

Handles end-to-end prediction:

```python
pp = PredictionPipeline("models/credit_risk_model.pkl")
predictions, probabilities = pp.make_prediction(df)
```

**Methods:**
- `make_prediction(df)`: Returns predictions and probabilities for input DataFrame

## 📁 Dataset Information

The dataset contains the following features:

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male/Female |
| Married | Categorical | Yes/No |
| Dependents | Categorical | Number of dependents (0, 1, 2, 3+) |
| Education | Categorical | Graduate/Not Graduate |
| Self_Employed | Categorical | Yes/No |
| ApplicantIncome | Numerical | Applicant's income |
| CoapplicantIncome | Numerical | Co-applicant's income |
| LoanAmount | Numerical | Loan amount requested |
| Loan_Amount_Term | Numerical | Loan term in months |
| Credit_History | Binary | Credit history (0/1) |
| Property_Area | Categorical | Urban/Semiurban/Rural |
| Loan_Status | Target | Loan approval status (Y/N) |

## 🛠️ Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **xgboost**: Gradient boosting framework
- **streamlit**: Web application framework
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **joblib**: Model persistence
- **shap**: Model interpretability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support, email [dhruvparmar70441.com] or create an issue in the repository.

---

**Note**: This is a machine learning project for educational and demonstration purposes. Always validate model performance on unseen data and consider business requirements when making real-world loan decisions.
