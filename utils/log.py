import logging

from utils.args import parse_arguments


def get_logger():
    """
    Get a logger instance for the application. If the logger already has
    handlers, it returns the existing logger.
    """

    logger = logging.getLogger(__name__)
    _, _, _, _, debug, logfile, _, _ = parse_arguments()

    if not logger.hasHandlers():
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
        )

        if logfile:
            handler = logging.FileHandler(logfile)
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    if debug is True:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger


def get_fileno():
    logger = get_logger()
    handle = logger.handlers[0]

    return handle.stream.fileno()
