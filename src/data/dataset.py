"""Dataset Manager"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import argparse
import pandas as pd

from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import Def
from src.utils.utilities import UtilityManager


class DatasetManager:
    """Dataset Manager"""

    def __init__(
            self,
            path: str,
            target: str,
            df: pd.DataFrame = None,
            is_inference: bool = False
    ) -> None:
        """Initialize dataset manager

        Args:
            path (str): Path to the dataset.
            target (str): Target feature from the dataset to predict.
            df (pd.DataFrame, optional): Loaded dataset. Defaults to None.
            is_inference (bool, optional): Whether is inference process or not. Defaults to False.
            
        Returns:
            None
        """
        self.path = path
        self.target = target
        self.df = df
        self.is_inference = is_inference
        
        self.train_data = None
        self.test_data = None
        
        self.is_loaded = self.df is not None
        self.is_processed = False
        
        self.relevant_features = list(Def.Data.VALIDATOR.keys())
        self.categorical_features = [col for col, info in Def.Data.VALIDATOR.items() if info['type'] == 'categorical']
        self.numerical_features = [col for col, info in Def.Data.VALIDATOR.items() if info['type'] == 'numerical']
        self.binary_features = [col for col, info in Def.Data.VALIDATOR.items() if info['type'] == 'logical']
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
        self.df = pd.read_csv(self.path, header=0)
        self.is_loaded = True
        return

    def select_features(self, features: list[str]):
        """Feature selection

        Args:
            features (list[str]): List of features to select

        Returns:
            None
        """
        if self.is_inference and self.target in features:
            features.remove(self.target)
            
        self.df = self.df[features]
        return

    def preprocess(self) -> None:
        """Preprocess features"""
        self.df['Mileage'] = self.df['Mileage'].str.replace(' km', '').astype(int)
        self.df['Engine volume'] = self.df['Engine volume'].str.replace(' Turbo', '').astype(float)
        return

    def add_features(self) -> None:
        """Exnted dataset with new features"""
        self.df['isTurbo'] = self.df['Engine volume'].apply(lambda x: 'Yes' if 'Turbo' in x else 'No')
        return

    def clean_data(self) -> None:
        """Data cleaning and validation"""
        # Make
        feature = 'Manufacturer'
        outliers_make = UtilityManager.Data.find_outliers_categorical(
            self.df,
            feature,
            min_freq=Def.Data.Param.MIN_FREQ_MAKE
        )
        self.df = self.df.drop(outliers_make.index).reset_index(drop=True)
        
        # Model
        feature = 'Model'
        outliers_model = UtilityManager.Data.find_outliers_categorical(
            self.df,
            feature,
            min_freq=Def.Data.Param.MIN_FREQ_MODEL
        )
        self.df = self.df.drop(outliers_model.index).reset_index(drop=True)
        
        # Production year
        feature = 'Prod. year'
        outliers_prodyear = UtilityManager.Data.find_outliers_numeric(
            self.df,
            feature=feature,
            iqr_threshold=Def.Data.Param.IQR_THRESHOLD_PROD_YEAR,
            min_value=Def.Data.VALIDATOR[feature]['min'],
            max_value=Def.Data.VALIDATOR[feature]['max'],
        )
        self.df = self.df.drop(outliers_prodyear.index).reset_index(drop=True)
        
        # Body category
        feature = 'Category'
        outliers_body = UtilityManager.Data.find_outliers_categorical(
            self.df,
            feature,
            min_freq=Def.Data.Param.MIN_FREQ_BODY_CATEGORY
        )
        self.df = self.df.drop(outliers_body.index).reset_index(drop=True)
        
        # Mileage
        feature = 'Mileage'
        outliers_mileage = UtilityManager.Data.find_outliers_numeric(
            self.df,
            feature=feature,
            iqr_threshold=Def.Data.Param.IQR_THRESHOLD_MILEAGE,
            min_value=Def.Data.VALIDATOR[feature]['min'],
            max_value=Def.Data.VALIDATOR[feature]['max'],
        )
        self.df = self.df.drop(outliers_mileage.index).reset_index(drop=True)
        
        # Car Price
        feature = 'Price'
        outliers_price = UtilityManager.Data.find_outliers_numeric(
            self.df,
            feature=feature,
            iqr_threshold=Def.Data.Param.IQR_THRESHOLD_PRICE,
            min_value=Def.Data.VALIDATOR[feature]['min'],
            max_value=Def.Data.VALIDATOR[feature]['max'],
        )
        self.df = self.df.drop(outliers_price.index).reset_index(drop=True)
        return

    def validate_data(self) -> None:
        """Validate data"""
        if self.is_inference and self.df.shape[0] != 1:
            raise ValueError("The instance should contain exactly one row for validation during the inference")

        for feature in self.df.columns:            
            value = self.df.iloc[0][feature]
            if not UtilityManager.Data.Validator.validate_feature_value(feature, value):
                raise ValueError(f"Invalid value '{value}' for feature '{feature}'")
        return

    def set_types(self) -> None:
        """Set types for features"""
        for feature in self.numerical_features:
            if self.is_inference and feature == self.target:
                continue
            self.df[feature] = self.df[feature].astype(float)

        self.df[self.categorical_features] = self.df[self.categorical_features].astype(str)

        self.df[self.binary_features] = self.df[self.binary_features].astype(str)
        return

    def execute_preparation(self, to_save: bool = False) -> pd.DataFrame:
        """Execute preparation of the raw dataset

        Returns:
            pd.DataFrame: Processed dataset
        """
        if not self.is_loaded:
            raise ValueError('Dataset is not loaded')

        if self.is_inference:
            # Validation
            self.validate_data()
        else:
            # Add features
            self.add_features()
            
            # Pre-process features
            self.preprocess()
            
            # Data cleaning
            self.clean_data()

        # Feature selection
        self.select_features(self.relevant_features)

        # Set types
        self.set_types()
        
        # Save
        if to_save:
            path = DatasetManager.get_processed_path()
            self.df.to_csv(path, index=False)
            logger.info(f'Saved processed data at: {path}')
        
        # Set flag
        self.is_processed = True
        
        return self.df

    def split(self, test_size: float = 0.2) -> None:
        """Split processed dataset into sets for training and testing

        Args:
            test_size (float, optional): Test size portion. Defaults to 0.2.
            
        Returns:
            None
        """
        self.train_data, self.test_data = train_test_split(self.df, test_size=test_size, random_state=Def.Env.SEED)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', '--download', action='store_true', help="Download the latest data")
    parser.add_argument('--process', '--process', action='store_true', help="Process data pipeline")

    args = parser.parse_args()

    # Download data
    if args.download:
        pass
    
    # Process data pipeline
    if args.process:
        manager = DatasetManager(
            path=DatasetManager.get_raw_path(),
            target='Price',
            df=None,
            is_inference=False
        )
        manager.load()
        manager.execute_preparation(to_save=True)
