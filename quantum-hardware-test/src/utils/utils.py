import random
import time
import logging
from datetime import datetime
from pathlib import Path

def setup_logger():
    """Set up the logger with file and console handlers."""
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('cryocooler_test')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = f'logs/cryocooler_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create a global logger instance
logger = setup_logger()

def simple_logger(message):
    """Simple logging function that logs to both file and console."""
    logger.info(message)

def inject_random_error(probability):
    return random.random() < probability
