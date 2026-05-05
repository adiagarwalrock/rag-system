import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = "rag.log"
_MAX_LOG_FILE_BYTES = 10 * 1024 * 1024
_LOG_BACKUP_COUNT = 5


def _log_file_path(filename: str) -> Path:
    return LOG_DIR / filename


def _has_file_handler(logger: logging.Logger, file_path: Path) -> bool:
    return _get_file_handler(logger, file_path) is not None


def _get_file_handler(
    logger: logging.Logger, file_path: Path
) -> logging.FileHandler | None:
    target = str(file_path.resolve())
    for handler in logger.handlers:
        handler_path = getattr(handler, "baseFilename", None)
        if not handler_path:
            continue
        if str(Path(handler_path).resolve()) == target:
            return handler
    return None


def configure_logging(
    *,
    log_level: int = logging.INFO,
    log_filename: str = DEFAULT_LOG_FILE,
) -> Path:
    """
    Configure application logging for console + rotating file output.
    This is safe to call repeatedly from multiple entrypoints.
    """
    log_file = _log_file_path(log_filename)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    file_handler = _get_file_handler(root_logger, log_file)
    if file_handler is None:
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=_MAX_LOG_FILE_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    file_handler.setLevel(log_level)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )
    if not has_stream_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Uvicorn and Streamlit can run with non-propagating loggers. Attach the
    # same file sink directly so API access/error logs are persisted.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "streamlit"):
        framework_logger = logging.getLogger(logger_name)
        if framework_logger.propagate:
            continue
        if _has_file_handler(framework_logger, log_file):
            continue
        framework_logger.addHandler(file_handler)

    return log_file
