"""Database"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import io
import json
import boto3 as aws
import pandas as pd

from typing import Any
from datetime import datetime
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
        
        self.session = aws.Session(profile_name=self.profile) if self.profile and Def.Env.IS_LOCAL else aws.Session()
        
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

    def save_version(self, content: dict, timestamp: str) -> bool:
        """Save version information in the storage

        Args:
            content (dict): Content about version to save
            timestamp (str): Timestamp to be used as file name

        Returns:
            bool: True if predictions are successfully saved, otherwise False
        """
        try:
            file_name = f'{timestamp}.json'
            to_path = os.path.join(Def.DB.VERSION_DIR, file_name)
            body = json.dumps(content)
            
            obj = self.conn.Object(self.bucket, to_path)
            obj.put(Body=body, **self.extra_args)
            obj.wait_until_exists()

            logger.info(f'Saved version info to {self.bucket}/{to_path}')
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
            logger.info(f"Found the latest dataset at {latest_file.key}")
                        
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
            logger.info(f"Copied data from {src_key} to {dest_key}")
            return True

        except Exception as ex:
            logger.error(f"Error copying file: {ex}")
            return False


    def get_object(self, path: str) -> Any:
        """Retrieve object from the given path

        Args:
            path (str): Path to the object

        Returns:
            Any: Object
        """
        response = self.client.get_object(Bucket=self.bucket, Key=path)
        
        obj = response['Body'].read().decode('utf-8')
        logger.info(f"Dataset downloaded from {self.bucket}/{path}")
        
        return obj


    def get_dataframe_from_csv(self, path: str) -> Any:
        """Retrieve object from the given path

        Args:
            path (str): Path to the object

        Returns:
            Any: Object
        """
        content = self.get_object(path=path)
        raw_data = io.StringIO(content)
        
        df = pd.read_csv(raw_data, header=0)
        logger.info(f"Dataset downloaded from {self.bucket}/{path}")
        
        return df

    def upload_dataframe_as_csv(self, path: str, df: pd.DataFrame) -> None:
        """Uploads an object to an S3 bucket.

        Args:
            path (str): Where the object will be stored in the bucket.
            df (pd.DataFrame): The data to upload. Can be a string or bytes.
            
        Returns:
            None
        """
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        body = csv_buffer.getvalue()
        
        self.client.put_object(
            Bucket=self.bucket,
            Key=path,
            Body=body,
            ContentType='text/csv'
        )
        
        logger.info(f"Dataset uploaded to {self.bucket}/{path}")
        return


    def upload_model(self, local_path: str, remote_dir: str) -> str:
        """Upload local model from temp directory into remote S3 directory

        Args:
            local_dir (str): Local temporary directory
            remote_dir (str): Remote S3 directory
            
        Returns:
            str: Path to the uploaded model
        """
        model_name = os.path.basename(local_path)
        remote_path = os.path.join(remote_dir, model_name)
        
        self.client.upload_file(local_path, self.bucket, remote_path)        
        logger.info(Def.Label.Model.UPLOADED_SUCCESSFULLY)
        
        return remote_path

    def download_model(self, remote_dir: str, local_dir: str) -> str:
        """Download the latest model file from the remote directory to the local directory.

        Args:
            remote_dir (str): Remote directory on S3.
            local_dir (str): Local directory.

        Returns:
            str: Path to the latest downloaded model
        """
        latest_model_file = self.find_latest_file(prefix=remote_dir)
        if latest_model_file is None:
            raise ValueError('No files found in the remote directory.')
        
        local_path = os.path.join(local_dir, os.path.basename(latest_model_file))
        local_dir_path = os.path.dirname(local_path)
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path)

        self.client.download_file(self.bucket, latest_model_file, local_path)
        logger.info(f'{Def.Label.Model.DOWNLOADED_SUCCESSFULLY}: {local_path}')
        
        return local_path

    def get_latest_model_dir(self, base_path: str) -> str:
        """Find the latest model directory based on timestamp

        Args:
            base_path (str): Base path to find the latest model

        Returns:
            str: Latest model path
        """
        folders = set()
        paginator = self.client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=f'{base_path}/', Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix_info in page['CommonPrefixes']:
                    folder_name = prefix_info['Prefix'].rstrip('/').split('/')[-1]
                    folders.add(folder_name)

        sorted_folders = sorted(list(folders))
        latest_dir = sorted_folders[-1]

        return latest_dir
