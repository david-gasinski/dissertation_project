from app import config
from flask import Blueprint, jsonify, request
import os

runs = Blueprint(
    "runs", 
    __name__,
    url_prefix='/runs'
)

@runs.route('/get', methods='get')
def get():
    path = config['ea']['runs_path']
    
    # return a list of runs 
    folder_names = []
    folder_paths = []
    
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            folder_names.append(name)
            folder_paths.append(full_path)
    
    return jsonify({
        "runs": folder_names,
        "paths": folder_paths
    })

@runs.route('/get_run', methods='post')
def get_run():
    # returns a list of generations with the 10 highest individuals from each
    request_data = request.get_json()['path'] 
    
    