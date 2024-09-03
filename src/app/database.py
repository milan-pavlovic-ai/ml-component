"""Database"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import uuid
import time as t
import boto3 as aws

from src.config import Def, logger


class StorageS3:
    """API for AWS S3 storage"""
    
    def __init__(self, bucket: str, region: str) -> None:
        """Initialize S3 storage

        Args:
            * bucket (str): AWS S3 bucket name
            * region (str): AWS region name
            
        Returns:
            None
        """
        self.bucket = bucket
        self.region = region
        self.conn = aws.resource('s3')
        self.client = aws.client('s3')
        self.extra_args = {
            'StorageClass': 'STANDARD'
        }
        return

    def clean_bucket(self) -> None:
        """Delete all items in the bucket"""
        try:
            self.bucket.objects.all().delete()
            logger.warning(Def.Label.Storage.CLEANED_BUCKET)
            return 0
        
        except Exception as err:
            logger.error(err)

        return Def.DB.EMPTY

    def save_response(self, preds: dict) -> bool:
        """Save prediction in storage

        Args:
            preds (dict): JSON object of predictions

        Returns:
            bool: True if predictions are successfully saved, otherwise False
        """
        try:
            file_name = f'{int(t.time())}_{uuid.uuid4()}.json'
            to_path = os.path.join(Def.DB.RESPONSE_DIR, file_name)
            
            obj = self.conn.Object(self.bucket, to_path)
            obj.put(Body=preds, **self.extra_args)
            obj.wait_until_exists()
 
            logger.info(f'Saved predictions to S3 bucket {self.bucket}/{Def.DB.RESPONSE_DIR}')
            return True

        except Exception as err:
            message = Def.Label.Storage.PREDS_NOT_SAVED
            logger.error(message)
            logger.error(err)
            raise ValueError(message)
