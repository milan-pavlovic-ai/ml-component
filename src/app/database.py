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
    
    def __init__(self, bucket: str, region: str, profile: str = None) -> None:
        """Initialize S3 storage

        Args:
            * bucket (str): AWS S3 bucket name
            * region (str): AWS region name
            * profile (str): AWS Profile name
            
        Returns:
            None
        """
        self.bucket = bucket
        self.region = region
        self.profile = profile
        
        self.session = aws.Session(profile_name=self.profile) if self.profile else aws.Session()
        
        self.conn = self.session.resource('s3', region_name=self.region)
        self.client = self.session.client('s3', region_name=self.region)
        
        self.extra_args = {'StorageClass': 'STANDARD'}
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

    def find_latest_file(self, prefix: str) -> str:
        """Find the latest file in the given path within the bucket.

        Args:
            * prefix (str): The path prefix to search for files
            
        Returns:
            * str: The key of the latest modified file
        """
        try:
            bucket = self.conn.Bucket(self.bucket)
            files = list(bucket.objects.filter(Prefix=prefix))
            
            if not files:
                return None
            
            latest_file = max(files, key=lambda x: x.last_modified)
            return latest_file.key

        except Exception as ex:
            logger.error(f"Error finding latest file: {ex}")
            return None
        
    def copy_file(self, src_key: str, dest_key: str) -> bool:
        """Copy a file to another location with.

        Args:
            * src_key (str): The key of the source file to copy
            * dest_key (str): The key of the destination file
            
        Returns:
            * bool: True if the copy operation was successful, False otherwise
        """
        try:
            copy_source = {
                'Bucket': self.bucket,
                'Key': src_key
            }
            self.client.copy(copy_source, self.bucket, dest_key, ExtraArgs=self.extra_args)
            return True

        except Exception as ex:
            logger.error(f"Error copying file: {ex}")
            return False
