"""Model"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import pandas as pd

from loguru import logger
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import Def
from src.data.dataset import DatasetManager


class PricingModel:
    """Pricing Model"""
    
    def __init__(self, dataset: DatasetManager) -> None:
        """Initialize model with dataset

        Args:
            dataset (DatasetManager): Dataset manager object
            
        Returns:
            None
        """
        self.dataset = dataset
        
        self.predictor = None
        self.train_data = None
        self.test_data = None
        return
        
    def save(self, path: str) -> None:
        """Save model at the given path

        Args:
            path (str): Path to save model
            
        Returns:
            None
        """
        self.predictor.save(path)
        return
    
    def load(self, path: str) -> None:
        """Load predictor at given path

        Args:
            path (str): Model path
            
        Returns:
            None
        """
        self.predictor = TabularPredictor.load(path)
        return

    def train(self, test_portion: float = 0.1) -> None:
        """Train model

        Args:
            test_portion (float): Portion of the dataset for testing. Defaults is 0.1.
            
        Returns:
            None
        """
        # Initialize
        self.predictor = TabularPredictor(
            label=self.dataset.target,
            eval_metric='mean_squared_error'
        )

        # Split dataset
        self.dataset.split(test_size=test_portion)
        
        # Fit predictor
        self.predictor.fit(self.dataset.train_data)
        return

    def eval(self) -> None:
        """Evaluate model on test dataset"""
        if self.predictor is None:
            raise ValueError(Def.Label.Model.NOT_LOADED_OR_TRAINED)
        
        # Make predictions on the test set
        y_pred = self.predictor.predict(self.dataset.test_data)

        # Calculate metrics
        y_true = self.dataset.test_data[self.dataset.target]
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Results
        logger.info(f"Mean Squared Error: {mse:.2f}")
        logger.info(f"Mean Absolute Error: {mae:.2f}")
        logger.info(f"R2 Score: {r2:.2f}")
        return

    def predict(self, input_data: pd.DataFrame) -> list[int]:
        """Predict car price for given cars

        Args:
            input_data (pd.DataFrame): Car instances
            
        Returns:
            (list[int]): Predicted car prices
        """
        if self.predictor is None:
            raise ValueError(Def.Label.Model.NOT_LOADED_OR_TRAINED)
        
        # Data workflow
        input_dataset = DatasetManager(df=input_data, target=None)
        input_dataset.execute_preparation()
        input_df = input_dataset.df
        
        # Predictions
        y_preds = self.predictor.predict(input_df)
        y_preds = y_preds.round().astype(int)
        
        return y_preds
