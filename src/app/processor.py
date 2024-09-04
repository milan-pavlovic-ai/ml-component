"""Data Processor"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import json

from loguru import logger
from typing import Dict, Any
from http import HTTPStatus

from src.config import Def
from src.app.database import StorageS3
from src.data.dataset import DatasetManager


# STORAGE

storage = StorageS3(
    bucket=Def.DB.S3_BUCKET,
    region=Def.Host.REGION,
    profile=Def.Host.PROFILE
)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda code for model training

    Args:
        event (Dict[str, Any]): Event
        context (Any): Context

    Returns:
        Dict[str, Any]: Response
    """
    try:
        # Find the latest dataset
        latest_dataset_path = storage.find_latest_file(prefix='data/raw')    
        latest_df = storage.get_dataframe_from_csv(path=latest_dataset_path)

        # Process dataset
        dataset = DatasetManager(path='s3-storage', target='Price', df=latest_df, is_inference=False)
        dataset.execute_preparation(to_save=False)
        df_processed = dataset.df

        # Save dataset
        dataset_name = latest_dataset_path.split('/')[-1][:-4]
        dataset_path = f'data/processed/{dataset_name}_processed.csv'
        storage.upload_dataframe_as_csv(path=dataset_path, df=df_processed)

        # Response
        response = {
            'statusCode': HTTPStatus.OK,
            'body': json.dumps(Def.Label.DataProcessor.PROCESSED_SUCCESSFULLY)
        }
        
    except Exception as ex:
        logger.error(str(ex))
        response = {
            'statusCode': HTTPStatus.INTERNAL_SERVER_ERROR,
            'body': json.dumps(Def.Label.DataProcessor.PROCESSED_FAILED)
        }

    return response


# if __name__ == '__main__':
#     lambda_handler(event=None, context=None)
