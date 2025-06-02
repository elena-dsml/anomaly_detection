import logging
from logging.handlers import RotatingFileHandler
import os


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "streamlit_anomaly_app.log")

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name="anomaly_detection_app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s — %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
