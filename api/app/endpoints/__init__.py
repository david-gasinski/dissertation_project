from flask import Blueprint
from app import config
from .matches import matches

endpoints = Blueprint(
    "endpoints", 
    __name__, 
    url_prefix=f"/{config.dict['parent_endpoint']}", 
)

endpoints.register_blueprint(matches)



 