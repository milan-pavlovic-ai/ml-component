import os
import sys

from loguru import logger
from dotenv import dotenv_values


def load_config(env: str, root_dir: str) -> dict[str, str]:
    """Load variables from .env file without altering the environment
   
    Args:
        * env (str): Environment type. It should be 'cloud', 'local', or empty string.
        * root_dir (str): Root directory of the project

    Returns:
        dict[str, str]: Dictionary of variables from configuration file
        
    Source:
        - https://saurabh-kumar.com/python-dotenv/
    """
    try:
        # Find file
        path = find_env_file(env=env, root_dir=root_dir)
        if not path:
            return {}
        
        # Load configuration
        config = dotenv_values(path)
        logger.info('Configuration has been loaded.')
        return config
        
    except Exception as err:
        logger.error(err)
        logger.error('Unable to load configuration file')
        return {}

def find_env_file(env: str, root_dir: str) -> str:
    """Find .env file in the given root directory of the project

    Args:
        * env (str): Environment type. It should be 'cloud', 'local', or empty string.
        * root_dir (str): Root directory of the project

    Returns:
        str: Path of the .env file in the project
    """
    path = None
    env_context = env.strip().lower()
    
    # File name
    if env_context in ('cloud', 'main'):
        env_file = 'cfg_cloud.env'
    
    elif env_context in ('local', 'debug'):
        env_file = 'cfg_local.env'

    else:
        env_file = '.env'
    
    # Set path
    path = os.path.join(root_dir, env_file)

    # Does exist
    if not os.path.exists(path):
        logger.error('Unable to find .env configuration file')
        return None
    
    return path


# Path
ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

# Environment
ENVIRONMENT = load_config('.env', ROOT_DIR)['ENVIRONMENT']
config = load_config(ENVIRONMENT, ROOT_DIR)


class Def:
    """Definitions and Constants"""
    
    class Env:
        """Environment variables"""
        SEED = 21
        DEBUG = sys.gettrace() is not None

    class Host:
        """Host information"""
        NAME = config['HOSTNAME']
        PORT = int(config['PORT'])
        URL = config['HOST_URL']
        REGION = config['AWS_REGION']
        
    class DB:
        """Database variables"""
        S3_BUCKET = config['S3_BUCKET']
        
    class Data:
        """Data"""
        class Dir:
            """Directory"""
            MAIN = os.path.join(ROOT_DIR, 'data')
            RAW = os.path.join(MAIN, 'raw')
            PROCESSED = os.path.join(MAIN, 'processed')
            TEMP = os.path.join(MAIN, 'temp')
        
    class Model:
        """Models"""        
        class Dir:
            """Directory"""
            MAIN = os.path.join(ROOT_DIR, 'models')
            PATH = os.path.join(MAIN, config['MODEL_PRICE'], 'model.pkl')
        
    
    class Label:
        """Labels"""
        class API:
            ACCESS_DENIED = 'Authorization: Access denied'

        class Model:
            NOT_LOADED_OR_TRAINED = 'Model is not loaded or trained'
