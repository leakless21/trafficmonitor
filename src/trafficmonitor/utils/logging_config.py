import sys
from loguru import logger

def setup_logging(): 
    """
    Configures Loguru logger with a specific format.
    Includes process name for clarity in multiprocessing.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{process.name: <15}</cyan> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.info("Logger initialized from setup_logging function.") # Added more context here