### log all the executions of the program
import logging
import os
from datetime import datetime 

LOG_FILE = f"{datetime.now().strftime('%m_%d_Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),"logs", LOG_FILE)
os.makedirs(log_path, exist_ok=True) ## keep on appending the logs even if the file exists


LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format= "%(asctime)s [%(levelname)s] [%(filename)s] [%(lineno)d] [%(message)s]" ,
    level=logging.INFO
)


if __name__ == "__main__":
    logging.info("This is info message")
    logging.log(logging.INFO, "This is info message")