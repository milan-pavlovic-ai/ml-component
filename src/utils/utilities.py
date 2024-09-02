"""Utility Manager"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import pandas as pd

from loguru import logger


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
