"""Start application"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import threading
import uvicorn
import pandas as pd

from loguru import logger
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from http import HTTPStatus
from mangum import Mangum

from src.config import Def
from src.app.interface import CarInterface
from src.utils.utilities import UtilityManager
from src.pricing.model import PricingModel
from src.data.dataset import DatasetManager
from src.app.database import StorageS3


# API

app = FastAPI(
    title='ML Component for Car Pricing',
    description='Machine Learning Component for Predicting Car Prices',
    version="1.0"
)


# STORAGE

storage = StorageS3(
    bucket=Def.DB.S3_BUCKET,
    region=Def.Host.REGION,
    profile=Def.Host.PROFILE
)


# MODELS

model = None
model_version = None
model_lock = threading.Lock()


def load_model() -> PricingModel:
    """Get and load the latest remote model

    Returns:
        PricingModel: Latest loaded model
    """
    global model, model_version

    with model_lock:

        # Local model
        if Def.Env.IS_LOCAL:
            model = PricingModel(dataset=None)
            model.load(path=Def.Model.Dir.PATH)
            return model
    
        # Check does need update
        current_model_version = storage.find_latest_file(prefix='models')
        
        if model_version != current_model_version:
            model_version = current_model_version
            
            # Download model
            model_path = storage.download_model(remote_dir='models', local_dir=Def.Model.Dir.TEMP_MAIN)
            
            # Load model
            model = PricingModel(dataset=None)
            model.load(path=model_path)
    
    return model


# ENDPOINTS

@app.post('/')
def ping() -> JSONResponse:
    """Ping API
        
    Returns:
    
        JSONResponse: Message
    """
    response = UtilityManager.Response.json_response_ok(Def.Label.API.PING_SUCCESSFUL)
    return response


@app.post("/values")
def allowed_values(feature: str = None) -> JSONResponse:
    """Allowed values for given feature name.
        If the feature value is empty, it will return list of all feature names.

    Args:
        
        feature (str): Feature name

    Raises:
        
        HTTPException: If feature doesn't exist

    Returns:
        
        JSONResponse: List of allowed values for requested feature
    """
    # List of features
    if not feature:
        response = UtilityManager.Response.create_json_response(
            content=list(Def.Data.VALIDATOR.keys())
        )
        return response
    
    # Values per feature
    if feature in Def.Data.VALIDATOR:
        response = UtilityManager.Response.create_json_response(
            content=Def.Data.VALIDATOR[feature]
        )
    else:
        response = UtilityManager.Response.json_response_err(
            message=f"Feature '{feature}' not found",
            status=HTTPStatus.BAD_REQUEST
        )
        
    return response


@app.post('/pricing')
def car_pricing(request: CarInterface) -> JSONResponse:
    """Predict car price for given input features
    
    Args:
    
        request (CarInterface): Input features of the car
        
    Returns:
    
        JSONResponse: Price of the car
    """
    try:
        # Load the latest model
        model = load_model()
        
        # Prepare request
        car_data = request.model_dump(by_alias=True)
        input_df = pd.DataFrame([car_data]) 
        
        # Prepare data
        dataset = DatasetManager(path='inference', target='Price', df=input_df, is_inference=True)
        dataset.execute_preparation(to_save=False)
        
        # Predict with model
        predictions = model.predict(input_data=dataset.df)
        
        # Create response
        content = {
            'carPrice': predictions.tolist(),
            'modelVersion': model_version
        }
        response = UtilityManager.Response.create_json_response(content=content)
    
    except ValueError as ex:
        message = str(ex)
        logger.error(Def.Label.API.PRICING_INVALID_INSTANCE + '\n' + message)
        response = UtilityManager.Response.json_response_err(
            message=message, status=HTTPStatus.BAD_REQUEST
        )
    
    except Exception as ex:
        logger.error(str(ex))
        response = UtilityManager.Response.json_response_err(
            message=Def.Label.API.PRICING_PREDICTION_FAILED, status=HTTPStatus.INTERNAL_SERVER_ERROR
        )
    
    return response


@app.post('/train_job')
def create_training_job() -> JSONResponse:
    """Create a training job for the model with the latest dataset version
    
    Returns:
    
        JSONResponse: Price of the car
    """    
    # Create a copy of the latest dataset version
    # In real-world scenario, this will extend the current dataset with the most recent car transactions
    
    # Find the latest
    current_dataset_path = storage.find_latest_file(prefix='data/raw')
    
    if current_dataset_path is None:
        response = UtilityManager.Response.json_response_err(
            message=Def.Label.API.TRAINING_JOB_FAILED, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        return response
    
    # Copy new version
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    new_dataset_path = f'{current_dataset_path.split("_")[0]}_{timestamp}.csv' if '_' in current_dataset_path else f'{current_dataset_path[:-4]}_{timestamp}.csv'
    creation_status = storage.copy_file(current_dataset_path, new_dataset_path)
    
    # Create response
    if creation_status:
        response = UtilityManager.Response.json_response_ok(
            message=Def.Label.API.TRAINING_JOB_SUCCESSFUL)
    else:
        response = UtilityManager.Response.json_response_err(
            message=Def.Label.API.TRAINING_JOB_FAILED, status=HTTPStatus.INTERNAL_SERVER_ERROR)
    
    return response


# AWS Gateway API
handler = Mangum(app)


# Debugging
if Def.Env.DEBUG:
    uvicorn.run(
        app=app,
        host=Def.Host.NAME,
        port=Def.Host.PORT
    )
