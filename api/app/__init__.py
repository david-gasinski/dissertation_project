import logging
import sys

from flask_cors import CORS
from flask import Flask

from configs.config import ConfigHandler
from logs.log import LogUtil

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

config = ConfigHandler.load_config("flask")
config_cors = config.dict["cors"]

# clear old logs if specified in config
if config.dict["clear_logs"]:
    print("Clearing logs...")
    LogUtil.clear_logs()

# capture fantasy logger, output to both
# stderr and file
logger = logging.getLogger("fantasy")
logger.setLevel(logging.INFO)

fantasy_log_path = LogUtil.create_log_file(config.dict["logs"]["fantasy"])
logger.addHandler(logging.FileHandler(fantasy_log_path, encoding='utf-8'))
logger.addHandler(logging.StreamHandler(sys.stderr))

# capture werkzeug output and direct it to flask specific log files
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_log_path = LogUtil.create_log_file(config.dict["logs"]["werkzeug"])
werkzeug_logger.addHandler(logging.FileHandler(werkzeug_log_path, encoding='utf-8'))

# define rate limiter
limiter_settings = config.dict['limiter'][config.dict['limiter']['storage_backend']]
limiter = Limiter(get_remote_address, **limiter_settings)

def create_app() -> Flask:
    from app.endpoints import endpoints

    logger.info("Initialising app...")

    app = Flask(__name__)
    app.debug = config.dict["debug"]
    app.config["SECRET_KEY"] = "fns-old-man"

    # CORS config
    CORS(app, **config_cors)

    logger.info("Loading endpoints...")
    app.register_blueprint(endpoints)

    return app
