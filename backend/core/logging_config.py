import logging


class ContextFilter(logging.Filter):
    """Add request/operation context to all log records."""

    def __init__(self):
        super().__init__()
        self.context = {}

    def filter(self, record):
        record.context = self.context
        return True


def setup_logging():
    context_filter = ContextFilter()

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(funcName)s:%(lineno)d - [%(context)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        format="%(asctime)s - %(name)s - %(levelname)s - "
               "%(funcName)s:%(lineno)d - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.addFilter(context_filter)

    return logger