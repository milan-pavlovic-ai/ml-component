"""API Interface"""

from pydantic import BaseModel, Field


class CarInterface(BaseModel):
    """
    Car Interface
    """
    manufacturer: str = Field(alias="Manufacturer")
    model: str = Field(alias="Model")
    prod_year: int = Field(alias="Prod. year")
    body_category: str = Field(alias="Category")
    mileage: int = Field(alias="Mileage")
    fuel_type: str = Field(alias="Fuel type")
    engine_volume: float = Field(alias="Engine volume")
    cylinders: int = Field(alias="Cylinders")
    gear_box: str = Field(alias="Gear box type")
    drive_wheels: str = Field(alias="Drive wheels")
    wheel_type: str = Field(alias="Wheel")
    color: str = Field(alias="Color")
    airbags: int = Field(alias="Airbags")
    leather_interior: bool = Field(alias="Leather interior")
