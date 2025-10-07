import yaml
from typing import Optional
from pathlib import Path
import logging

from .schema import AppConfig

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    if config_path is None:
        config_path = Path("configs/config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = AppConfig(**config_dict)
        
        logger.info(f"Config loaded successfully from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise ValueError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        logger.error(f"Config loading error: {e}")
        raise