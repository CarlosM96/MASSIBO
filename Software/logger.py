import os
import logging

LOG_TO_FILE=True
LOG_FILENAME=".log"

logger = logging.getLogger("daphne")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)