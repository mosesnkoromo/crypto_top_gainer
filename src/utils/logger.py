"""src/utils/logger.py"""
import logging, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _EATFormatter(logging.Formatter):
    """Logging formatter that displays timestamps in East Africa Time (UTC+3)."""
    import datetime
    from zoneinfo import ZoneInfo
    _EAT = ZoneInfo("Africa/Dar_es_Salaam")

    def formatTime(self, record, datefmt=None):
        import datetime
        from zoneinfo import ZoneInfo
        eat = ZoneInfo("Africa/Dar_es_Salaam")
        dt = datetime.datetime.fromtimestamp(record.created, tz=eat)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S EAT")


def setup_logging(log_dir: Path, filename: str, level: str, max_bytes: int, backup_count: int):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = _EATFormatter("%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    fh = RotatingFileHandler(log_dir/filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(fmt); root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)