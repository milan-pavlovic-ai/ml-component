"""Model Training"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import json
import traceback

from loguru import logger
from typing import Dict, Any
from http import HTTPStatus

from src.config import Def
from src.app.database import StorageS3
from src.data.dataset import DatasetManager
from src.pricing.model import PricingModel


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
        latest_dataset_path = storage.find_latest_file(prefix='data/processed')    
        latest_df = storage.get_dataframe_from_csv(path=latest_dataset_path)
    
        # Prepare dataset
        dataset = DatasetManager(
            path=latest_dataset_path,
            target='Price',
            df=latest_df,
            is_inference=False
        )
        dataset.split()
    
        # Train model
        model = PricingModel(dataset=dataset)
        model.train()

        # Eval model
        results = model.eval()

        # Upload model
        storage.upload_model(local_path=model.model_path, remote_dir='models')

        # Response
        content = {}
        content['dataset'] = latest_dataset_path
        content['model'] = model.model_path
        content['metrics'] = results
        storage.save_version(content=content, timestamp=model.version)
        
        response = {
            'statusCode': HTTPStatus.OK,
            'body': json.dumps(content)
        }
        
    except Exception as ex:
        logger.error(str(ex))
        logger.error(traceback.format_exc())
        response = {
            'statusCode': HTTPStatus.INTERNAL_SERVER_ERROR,
            'body': json.dumps(Def.Label.DataProcessor.PROCESSED_FAILED)
        }

    return response


# if __name__ == '__main__':
#     response = lambda_handler(event=None, context=None)
#     print(response)
    