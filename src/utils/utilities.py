"""Utility Manager"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import pandas as pd

from typing import Union
from loguru import logger
from http import HTTPStatus
from fastapi.responses import JSONResponse

from src.config import Def, logger


class UtilityManager:
    """Utility Manager"""
    
    def __init__(self) -> None:
        """Constructor"""
        return
    
    class Data:
        """Data Utilities"""
        
        def find_outliers_numeric(
                df: pd.DataFrame,
                feature: str,
                iqr_threshold: float,
                min_value: float,
                max_value: float
        ) -> pd.DataFrame:
            """Find outliers from the dataset based on given feature and paremeters

            Args:
                df (pd.DataFrame): Dataset
                feature (str): Feature name
                iqr_threshold (float): IQR threshold value
                min_value (float): Min value of feature
                max_value (float): Max value of feature

            Returns:
                pd.DataFrame: Outliers
            """
            # Identify bounds
            values = df[feature]
            q25 = int(round(values.quantile(0.25)))
            q75 = int(round(values.quantile(0.75)))
            iqr_value = q75 - q25
            
            lower_bound = q25 - iqr_threshold * iqr_value
            if lower_bound < min_value:
                lower_bound = min_value

            upper_bound = q75 + iqr_threshold * iqr_value
            if upper_bound > max_value:
                upper_bound = max_value

            logger.info(f'{lower_bound} < {feature} < {upper_bound}')
            
            # Identify outliers
            mask_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
            outliers = df[mask_outliers]
            
            total_data_points = len(df)
            num_outliers = len(outliers)
            percentage_outliers = (num_outliers / total_data_points) * 100
            
            logger.info(f'Outlier for {feature}: {num_outliers} or {percentage_outliers:.2f}%')
            logger.info(f'Before {total_data_points}, After {total_data_points - num_outliers}')
            
            return outliers
                
        def find_outliers_categorical(df: pd.DataFrame, feature: str, min_freq: int) -> pd.DataFrame:
            """Find outliers for categorical feature under given minimum frequency

            Args:
                df (pd.DataFrame): Dataset
                feature (str): Feature name
                min_freq (int): Minimum frequency

            Returns:
                pd.DataFrame: Outliers
            """
            counts = df[feature].value_counts()
            
            mask = df[feature].isin(counts[counts <= min_freq].index)
            outliers = df[mask]

            total_data_points = len(df)
            num_outliers = len(outliers)
            percentage_outliers = (num_outliers / total_data_points) * 100
            
            logger.info(f'Outlier for {feature}: {num_outliers} or {percentage_outliers:.2f}%')
            
            return outliers

        def calc_stats(df, feature):
            """Calculate statistics for given feature in the dataset"""
            values = df[feature]
            logger.info(f'\nStatistics for {feature}')
            logger.info(f'Mean = {values.mean():.2f}')
            logger.info(f'Std = {values.std():.2f}')
            logger.info(f'Min = {values.min():.2f}')
            logger.info(f'25th = {values.quantile(0.25):.2f}')
            logger.info(f'50th = {values.median():.2f}')
            logger.info(f'75th = {values.quantile(0.75):.2f}')
            logger.info(f'Max = {values.max():.2f}')
            logger.info(f'IQR = {(values.quantile(0.75) - values.quantile(0.25)):.2f}')
            return

        class Validator:
            """Validator utilities"""
            
            def validate_feature_value(feature: str, value: Union[str, int, float]) -> bool:
                """Validate feature value

                Args:
                    feature (str): Feature name
                    value (Union[str, int, float]): Feature value

                Raises:
                    ValueError: If feature is not in the Validator

                Returns:
                    bool: True if feature value is valid, otherwise False
                """
                if feature not in Def.Data.VALIDATOR:
                    raise ValueError(f"Feature '{feature}' is not valid")

                feature_info = Def.Data.VALIDATOR[feature]

                if feature_info["type"] == "categorical":
                    is_valid = value in feature_info["values"]
                
                elif feature_info["type"] == "numerical":
                    is_valid = (feature_info["min"] <= value <= feature_info["max"])
                
                else:
                    is_valid = False
                    
                return is_valid

    class Response:
        """Response utilites"""

        @staticmethod
        def json_response_err(message: str, status: HTTPStatus) -> JSONResponse:
            """
            Create json response with error flag
            Args:
                message (str): content of message
                status (HTTPStatus): status of message
            Returns:
                JSONResponse: response in json format for given message
            """
            if status is None:
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                
            return JSONResponse(content={'error': message}, status_code=status)
        
        @staticmethod
        def json_response_ok(message: str, status: HTTPStatus=HTTPStatus.OK) -> JSONResponse:
            """
            Create valid json response with given message
                Args:
                    message (str): content of message
                    status (HTTPStatus): status of message. Defaults HTTPStatus.OK.
                Returns:
                    JSONResponse: response in json format for given message
            """
            if status is None:
                status = HTTPStatus.OK
                
            return JSONResponse(content={'message': message}, status_code=status)

        @staticmethod
        def create_json_response(content: dict, status: HTTPStatus=HTTPStatus.OK) -> JSONResponse:
            """
            Create json response for given dictionary
                Args:
                    content (dict): content in dict form
                    status (HTTPStatus): status of message. Defaults HTTPStatus.OK.
                Returns:
                    JSONResponse: json response for given dictionary
            """
            if status is None:
                status = HTTPStatus.OK
                
            return JSONResponse(content=content, status_code=status)
