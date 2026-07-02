import logging
from datetime import datetime

def simple_logger(message: str):
    """
    Simple logging function that prints timestamped messages.
    
    Args:
        message (str): Message to log
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}") 