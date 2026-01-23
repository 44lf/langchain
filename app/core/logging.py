import logging
import os
from logging import Logger



def setup_logging() -> None:
    _DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    # Basic stdout logging; idempotent for repeated calls.
    logging.basicConfig(
        level=_DEFAULT_LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def get_logger(name: str) -> Logger:
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)


