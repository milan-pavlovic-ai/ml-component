"""Dataset Manager"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import pandas as pd

from src.config import Def
from src.utils.utilities import UtilityManager


class DatasetManager:
    """Dataset Manager"""
    
    RELEVANT_RAW_FEATURES = [
        'Manufacturer',
        'Model',
        'Prod. year',
        'Category',
        'Mileage',
        'Fuel type',
        'Engine volume',
        'Cylinders',
        'Gear box type',
        'Drive wheels',
        'Wheel',
        'Color',
        'Airbags',
        'Leather interior',
        'Price'
    ]
    
    RELEVANT_FEATURES = [
        'Manufacturer',
        'Model',
        'Prod. year',
        'Category',
        'Mileage',
        'Fuel type',
        'Engine volume',
        'Cylinders',
        'isTurbo',
        'Gear box type',
        'Drive wheels',
        'Wheel',
        'Color',
        'Airbags',
        'Leather interior',
        'Price'
    ]
    
    NUMERICAL_FEATURES = ['Prod. year', 'Mileage', 'Engine volume', 'Cylinders', 'Airbags', 'Price']
    
    CATEGORICAL_FEATURES = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels', 'isTrubo', 'Wheel', 'Color', 'Leather interior']
    
    def __init__(self, path: str, target: str) -> None:
        """Initialize dataset manager

        Args:
            path (str): Path to the dataset
            target (str): Target feature from the dataset to predict
            
        Returns:
            None
        """
        self.path = path
        self.target = target
        
        self.df_raw = None
        self.df = None
        
        self.is_loaded = False
        self.is_processed = False
        return

    @staticmethod
    def get_raw_path() -> str:
        """Returns path to the raw dataset

        Returns:
            str: Path of raw dataset
        """
        path = os.path.join(Def.Data.Dir.RAW, 'car-data.csv')
        return path

    @staticmethod
    def get_processed_path() -> str:
        """Returns path to the processed dataset

        Returns:
            str: Path of raw dataset
        """
        path = os.path.join(Def.Data.Dir.PROCESSED, 'car-data-processed.csv')
        return path

    def load(self) -> None:
        """Load dataset"""
        self.df_raw = pd.read_csv(self.path, header=0)
        self.is_loaded = True
        return

    def process(self) -> pd.DataFrame:
        """Pre-process raw dataset

        Returns:
            pd.DataFrame: Processed dataset
        """
        if not self.is_loaded:
            raise ValueError('Dataset is not loaded')
        
        # Feature selection
        self.df = self.df_raw[DatasetManager.RELEVANT_RAW_FEATURES]
        
        # Add features
        self.df['isTurbo'] = self.df['Engine volume'].apply(lambda x: 'Yes' if 'Turbo' in x else 'No')
        
        # Pre-process features
        self.df['Mileage'] = self.df['Mileage'].str.replace(' km', '').astype(int)
        self.df['Engine volume'] = self.df['Engine volume'].str.replace(' Turbo', '').astype(float)
        
        # Data cleaning and validation        
        feature = 'Manufacturer'
        outliers_make = UtilityManager.Data.find_outliers_categorical(self.df, feature, min_freq=5)
        self.df = self.df.drop(outliers_make.index).reset_index(drop=True)
        
        feature = 'Model'
        outliers_model = UtilityManager.Data.find_outliers_categorical(self.df, feature, min_freq=3)
        self.df = self.df.drop(outliers_model.index).reset_index(drop=True)
        
        feature = 'Prod. year'
        outliers_prodyear = UtilityManager.Data.find_outliers_numeric(self.df, feature=feature, iqr_threshhold=7, min_value=1950, max_value=2025)
        self.df = self.df.drop(outliers_prodyear.index).reset_index(drop=True)
        
        feature = 'Category'
        outliers_body = UtilityManager.Data.find_outliers_categorical(self.df, feature, min_freq=10)
        self.df = self.df.drop(outliers_body.index).reset_index(drop=True)
        
        feature = 'Mileage'
        outliers_mileage = UtilityManager.Data.find_outliers_numeric(self.df, feature=feature, iqr_threshhold=3, min_value=0, max_value=600_000)
        self.df = self.df.drop(outliers_mileage.index).reset_index(drop=True)
        
        feature = 'Price'
        outliers_price = UtilityManager.Data.find_outliers_numeric(self.df, feature=feature, iqr_threshhold=7.5, min_value=1000, max_value=150_000)
        self.df = self.df.drop(outliers_price.index).reset_index(drop=True)

        # Set types
        self.df[DatasetManager.NUMERICAL_FEATURES] = self.df[DatasetManager.NUMERICAL_FEATURES].astype(float)
        self.df[DatasetManager.CATEGORICAL_FEATURES] = self.df[DatasetManager.CATEGORICAL_FEATURES].astype(str)
        
        self.is_processed = True
        
        return self.df
