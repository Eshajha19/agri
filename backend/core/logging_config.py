import logging


class ContextFilter(logging.Filter):
    """Add request/operation context to all log records."""

    def __init__(self):
        super().__init__()
        self.context = {}

    def filter(self, record):
        # Only add context to the log record if context is not empty.
        # Prevents cluttering logs with unused context attributes.
        if self.context:
            record.context = self.context
        else:
            record.context = ""
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

    logger = logging.getLogger(__name__)
    logger.addFilter(context_filter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger