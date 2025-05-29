import multiprocessing
import time
import loguru
from .utils.logging_config import setup_logging

def main():
    setup_logging
    loguru.logger.info("Starting main supervisor process...")
    shutdown_event = multiprocessing.Event()
    
