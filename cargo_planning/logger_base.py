import logging
from logging.config import dictConfig


class CustomException(Exception):
    def __init__(self, message):
        self.message = message


def configure_logger(name, log_path):
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "level": "DEBUG",
                    "class": "logging.FileHandler",
                    "formatter": "default",
                    "filename": log_path,
                },
            },
            "loggers": {"default": {"level": "DEBUG", "handlers": ["console", "file"]}},
            "disable_existing_loggers": False,
        }
    )
    return logging.getLogger(name)


logger = configure_logger("default", "info.log")
