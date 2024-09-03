"""Start application"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from http import HTTPStatus
from mangum import Mangum

from src.config import Def
from src.app.interface import CarInterface, PingInteface
from src.utils.utilities import UtilityManager
from src.pricing.model import PricingModel


# API
app = FastAPI(
    title='ML Component for Car Pricing',
    description='Machine Learning Compoent for predicting car prices',
    version=1.0
)


# MODELS
model = PricingModel.load(path=Def.Model.Dir.PATH)


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


@app.post('/pricing')
def car_pricing(request: CarInterface) -> JSONResponse:
    """Predict car price for given input features
    
    Args:
    
        request (CarInterface): Input features of the car
        
    Returns:
    
        JSONResponse: Price of the car
    """
    # TODO
    
    message = 'Car Pricing'
    response = UtilityManager.Response.json_response_ok(message)
    
    return response



# AWS Gateway API
handler = Mangum(app)


# Debugging
if Def.Env.DEBUG:
    uvicorn.run(
        app=app,
        host=Def.Host.NAME,
        port=Def.Host.PORT,
        debug=Def.Env.DEBUG
    )
