import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.data_cleaning import DataCleaner

class FeatureProcessor:
    def __init__(self, df: pd.DataFrame):
        self.master_df = df.copy()
        
    def encode_categorical(self):
        cat_col = self.master_df.select_dtypes('object').columns
        for col in cat_col:
            encoder = LabelEncoder()
            self.master_df[col] = encoder.fit_transform(self.master_df[col])
        return self

    def scale_features(self):
        num_col = self.master_df.select_dtypes(np.number).columns
        scaler = StandardScaler()
        self.master_df[num_col] = scaler.fit_transform(self.master_df[num_col])  
        return self 
    
    def get_features(self):
        return self.master_df
