import os, sys, logging
from datetime import datetime


# Create a log filename with the current date and time
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Define the directory where log files will be stored
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)  #Creating the log directory if it doesn't exist

LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE) # Full path for the log file

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

# File handler - Create Logging in files
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(formatter)

# Stream (console) handler - Show realtime logs in terminal
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)