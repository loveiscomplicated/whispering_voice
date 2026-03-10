"""Logging utility for the STT data generation pipeline."""

import logging
import sys
from pathlib import Path


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Create and configure a logger with console (and optional file) output.

    Each call is idempotent: if a logger with *name* already has handlers
    attached, the existing logger is returned unchanged to avoid duplicate
    log entries.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_file: Optional path to a log file. The parent directory is
            created automatically if it does not exist. When ``None``, only
            console output is used.
        level: Logging level applied to both the logger and its handlers.
            Defaults to ``logging.DEBUG``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Return early if already configured to prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler (stdout for DEBUG/INFO, stderr for WARNING+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    if log_file is not None:
        _add_file_handler(logger, log_file, level)

    # Prevent log records from propagating to the root logger
    logger.propagate = False

    return logger


def _add_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: int,
) -> None:
    """Attach a rotating-friendly file handler to *logger*.

    Args:
        logger: Logger instance to attach the handler to.
        log_file: Destination file path.
        level: Logging level for the file handler.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)
