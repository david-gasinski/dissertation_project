import os
import json
from pathlib import Path
import logging
from typing import Any

logger = logging.getLogger('fantasy')
logger.setLevel(logging.ERROR)

class Config():
    
    def __init__(self, name: str, config: dict) -> None: 
        self.name = name
        self.dict = config
        
    def save_config(self) -> None:
        ConfigHandler.save_config(self.name, self.dict)
    
    def add_attr(self, key: str, value: Any) -> None:
        self.config[key] = value
        ConfigHandler.save_config(self.name, self.dict)
       
    def del_attr(self, key: str) -> None:
        self.config.pop(key, None)
        ConfigHandler.save_config(self.name,self.dict)

    
class ConfigHandler():
        
    @staticmethod
    def load_config(name: str) -> Config:
        """
        Returns a ```{name}``` config from ```{root}/configs/```
        
        Args:
            name (str): name of the config file to load
            
        Returns:
            Config: config object
        """

        path = ConfigHandler._get_full_config_path(name)
        
        try:
            with open(path, 'r') as content:
                config_dict = json.loads(content.read())
        except json.JSONDecodeError as e:
            logger.error(
                f"Could not decode JSON from {path}.",
                exc_info=e
            )
        except FileNotFoundError as e:
            logger.error(
                f"Could not locate {path}",
                exc_info=e
            )    

        return Config(name, config_dict)
    
    @staticmethod 
    def save_config(name: str, config: dict) -> None:
        """
        Saves config into ```{root}/configs/{name}.json```
        
        Args:
            name (str): Name of config file
            config (Config): 
        
        Returns:
            None
        """
        path = ConfigHandler._get_full_config_path(name)
        
        try:
            with open(path, 'w') as config_file:
                json.dump(config, config_file)
        except TypeError as e:
            logger.error(
                "Unable to serliaze json object",
                exc_info=e
            )
        except FileNotFoundError as e:
            logger.error(
                f"Config file {name} doesn't exist",
                exc_info=e
            )
    
    @staticmethod
    def _get_full_config_path(name: str) -> str:
        """
        Helper method. Returns a full path of the config file
            {root}/configs/{name}.json
            
        name: str
            Name of the config file
        
        return: str
        """ 
        if not name.endswith('.json'):
            name += '.json'
        return Path('configs', name).absolute()