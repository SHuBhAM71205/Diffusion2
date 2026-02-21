import os
import logging
import sys

def setup_logger(file="./logs/gen.log"):
    
    if not os.path.exists(file):
        os.makedirs(file,exist_ok=True)
    
    log_file = os.path.join(file)
    
    logger = logging.getLogger("Diffusion Logger")
    
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    # Handler 2: File
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    
    return logger