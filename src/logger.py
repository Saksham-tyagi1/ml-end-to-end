import logging
import os
from datetime import datetime

# Generate a log file name based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create a logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure the logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Create a logger
logger = logging.getLogger(__name__)
