"""
modules/logger.py
Structured, tagged logging for every pipeline stage.
Tags: [LLM] [FILTER] [SCORE] [SELECTED] [PIPELINE] [TTS] [CLIP] [SUB] [RENDER] [ERROR]
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Colour codes for terminal (stripped in file logs)
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}

class _TaggedFormatter(logging.Formatter):
    def __init__(self, use_colour=False):
        super().__init__()
        self.use_colour = use_colour

    def format(self, record):
        tag  = getattr(record, "tag", "PIPELINE")
        ts   = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        lvl  = record.levelname
        msg  = record.getMessage()

        line = f"[{ts}] [{tag:<8}] [{lvl:<7}] {msg}"

        if self.use_colour:
            c = _COLOURS.get(lvl, "")
            r = _COLOURS["RESET"]
            line = f"{c}{line}{r}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def _make_logger(log_dir: str, log_level: str, log_to_file: bool) -> logging.Logger:
    logger = logging.getLogger("shorts_engine")
    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, log_level.upper(), logging.DEBUG)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_TaggedFormatter(use_colour=True))
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fname = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        fh = logging.FileHandler(os.path.join(log_dir, fname), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(_TaggedFormatter(use_colour=False))
        logger.addHandler(fh)

    return logger


_logger: logging.Logger | None = None

def init_logger(log_dir="logs", log_level="DEBUG", log_to_file=True) -> logging.Logger:
    global _logger
    _logger = _make_logger(log_dir, log_level, log_to_file)
    return _logger

def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = _make_logger("logs", "DEBUG", True)
    return _logger

def _log(level: str, msg: str, tag: str, exc_info=False):
    lg = get_logger()
    extra = {"tag": tag}
    getattr(lg, level)(msg, extra=extra, exc_info=exc_info)

# Convenience helpers — use these everywhere
def log_llm(msg, level="info"):      _log(level, msg, "LLM")
def log_filter(msg, level="info"):   _log(level, msg, "FILTER")
def log_score(msg, level="info"):    _log(level, msg, "SCORE")
def log_selected(msg, level="info"): _log(level, msg, "SELECTED")
def log_pipeline(msg, level="info"): _log(level, msg, "PIPELINE")
def log_tts(msg, level="info"):      _log(level, msg, "TTS")
def log_clip(msg, level="info"):     _log(level, msg, "CLIP")
def log_sub(msg, level="info"):      _log(level, msg, "SUB")
def log_render(msg, level="info"):   _log(level, msg, "RENDER")
def log_error(msg, exc=False):       _log("error", msg, "ERROR", exc_info=exc)
def log_debug(msg, tag="PIPELINE"):  _log("debug", msg, tag)