"""Start application"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import uvicorn
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from http import HTTPStatus
from mangum import Mangum

from src.config import Def
from src.app.interface import CarInterface, PingInteface
from src.utils.utilities import UtilityManager
from src.pricing.model import PricingModel
from src.data.dataset import DatasetManager


# API

app = FastAPI(
    title='ML Component for Car Pricing',
    description='Machine Learning Compoent for predicting car prices',
    version="1.0"
)


# MODELS

model = PricingModel(dataset=None)
model.load(path=Def.Model.Dir.MAIN)


# ENDPOINTS

@app.post('/')
def ping(request: PingInteface) -> JSONResponse:
    """Ping API
    
    Args:
    
        message (PingInteface): Hello message
        
    Returns:
    
        JSONResponse: Message
    """
    if request.message:
        message = request.message
    else:
        message = Def.Label.API.PING_SUCCESSFUL
    
    response = UtilityManager.Response.json_response_ok(message)
    
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
    # Prepare request
    car_data = request.model_dump(by_alias=True)
    input_df = pd.DataFrame([car_data]) 
    
    # Prepare data
    dataset = DatasetManager(path='inference', target='Price', df=input_df, is_inference=True)
    dataset.execute_preparation(to_save=False)
    
    # Predict with model
    predictions = model.predict(input_data=dataset.df)
    
    # Create response
    content = {'carPrice': predictions.tolist()}
    response = UtilityManager.Response.create_json_response(content=content)
    
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
