import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.master_df = df

    def handle_missing_values(self):
        num_col = self.master_df.select_dtypes(np.number).columns
        cat_col = self.master_df.select_dtypes('object').columns

        for col in num_col:
            self.master_df[col] = self.master_df[col].fillna(self.master_df[col].median())

        
        for col in cat_col:
            self.master_df[col] = self.master_df[col].fillna(self.master_df[col].mode()[0])

        return self
    
    def remove_duplicates(self):
        self.master_df.drop_duplicates()
        return self


