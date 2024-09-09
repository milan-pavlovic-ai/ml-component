"""Model"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import pickle
import argparse
import pandas as pd

from loguru import logger
from datetime import datetime
from xgboost import XGBRegressor
from category_encoders import TargetEncoder, BinaryEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
        self.main_metric = 'mean_squared_error'
        
        self.predictor = None
        self.model_name = None
        self.model_path = None

        self.train_data = None
        self.test_data = None
        return
        
    def save(self, path: str) -> None:
        """Save model at the given path

        Args:
            path (str): Model path
            

        Returns:
            None
        """
        # Save model
        with open(path, 'wb') as file:
            pickle.dump(self.predictor, file)
        logger.info(f'Model saved to path: {path}')
        return
    
    def load(self, path: str) -> None:
        """Load predictor at given path

        Args:
            path (str): Model path
            
        Returns:
            None
        """
        with open(path, 'rb') as file:
            self.predictor = pickle.load(file)
        logger.info(f'Loaded model from path: {path}')
        return

    def train(self) -> None:
        """Train model"""
        # Pipelines
        self.predictor = Pipeline(steps=[
            ('cat_encoder', TargetEncoder(cols=self.dataset.categorical_features)),
            ('bin_encoder', BinaryEncoder(cols=self.dataset.binary_features)),
            ('scaler', StandardScaler()),
            ('estimator', XGBRegressor(
                objective='reg:squarederror',
                max_depth=10,
                random_state=Def.Env.SEED,
                verbosity=3
            ))
        ])
        
        # Fit predictor
        input_data = self.dataset.train_data.drop(columns=[self.dataset.target])
        target_data = self.dataset.train_data[self.dataset.target]
        
        self.predictor.fit(X=input_data, y=target_data)

        # Save
        self.model_name = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        model_dir = Def.Model.Dir.MAIN if Def.Env.IS_LOCAL else Def.Model.Dir.TEMP_MAIN
        self.model_path = os.path.join(model_dir, self.model_name)
        self.save(path=self.model_path)
        return

    def eval(self) -> dict[str, float]:
        """Evaluate model on test dataset

        Raises:
            ValueError: If the model is not trained

        Returns:
            dict[str, float]: Scores
        """
        if self.predictor is None:
            raise ValueError(Def.Label.Model.NOT_LOADED_OR_TRAINED)
        
        # Make predictions on the test set
        input_test_data = self.dataset.test_data.drop(columns=[self.dataset.target])
        y_true = self.dataset.test_data[self.dataset.target]
        
        y_pred = self.predictor.predict(input_test_data)

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Results
        results = {
            'Mean Squared Error': round(mse, 2),
            'Mean Absolute Error': round(mae, 2),
            'R2 Score': round(r2, 2)
        }
        logger.info(results)
        
        return results

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
        input_dataset = DatasetManager(path='run-time', target='Price', df=input_data, is_inference=True)
        input_dataset.execute_preparation()
        input_df = input_dataset.df
        
        # Predictions
        y_preds = self.predictor.predict(input_df)
        y_preds = y_preds.round().astype(int)
        
        return y_preds



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '--train', action='store_true', help="Train model")
    parser.add_argument('--eval', '--eval', action='store_true', help="Evaluate model")

    args = parser.parse_args()
    
    # Prepare dataset
    dataset = DatasetManager(
        path=DatasetManager.get_processed_path(),
        target='Price',
        df=None,
        is_inference=False
    )
    dataset.load()
    dataset.split()
        
    # Initialize model
    model = PricingModel(dataset=dataset)
    
    # Train model
    if args.train:
        model.train()
    
    # Evaluate model
    if args.eval:
        model.load(Def.Model.Dir.PATH)
        model.eval()
