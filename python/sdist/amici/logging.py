"""
Logging
-------
This module provides custom logging functionality for other amici modules
"""

import functools
import logging
import os
import platform
import socket
import time
import warnings
from inspect import currentframe, getouterframes

import amici

LOG_LEVEL_ENV_VAR = "AMICI_LOG"
BASE_LOGGER_NAME = "amici"
# Supported values for LOG_LEVEL_ENV_VAR
NAMED_LOG_LEVELS = {
    "NOTSET": logging.NOTSET,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

from typing import Callable, Optional, Union


def _setup_logger(
    level: Optional[int] = logging.WARNING,
    console_output: Optional[bool] = True,
    file_output: Optional[bool] = False,
    capture_warnings: Optional[bool] = False,
) -> logging.Logger:
    """
    Set up a new :class:`logging.Logger` for AMICI logging.

    :param level:
        Logging level, typically using a constant like :obj:`logging.INFO` or
        :obj:`logging.DEBUG`

    :param console_output:
        Set up a default console log handler if ``True`` (default)

    :param file_output:
        Supply a filename to copy all log output to that file, or
        set to ``False`` to disable (default)

    :param capture_warnings:
        Capture warnings from Python's warnings module if ``True``

    :return:
        A :class:`logging.Logger` object for AMICI logging. Note that other
        AMICI modules
        should use a logger specific to their namespace instead by calling
        :func:`get_logger`.
    """
    log = logging.getLogger(BASE_LOGGER_NAME)

    # Logging level can be overridden with environment variable
    if LOG_LEVEL_ENV_VAR in os.environ:
        try:
            level = int(os.environ[LOG_LEVEL_ENV_VAR])
        except ValueError:
            # Try parsing as a name
            level_name = os.environ[LOG_LEVEL_ENV_VAR]
            if level_name in NAMED_LOG_LEVELS.keys():
                level = NAMED_LOG_LEVELS[level_name]
            else:
                raise ValueError(
                    f"Environment variable {LOG_LEVEL_ENV_VAR} "
                    f'contains an invalid value "{level_name}".'
                    f" If set, its value must be one of "
                    f'{", ".join(NAMED_LOG_LEVELS.keys())}'
                    f" (case-sensitive) or an integer log level."
                )

    log.setLevel(level)

    py_warn_logger = logging.getLogger("py.warnings")

    # Remove default logging handler
    for handler in log.handlers:
        if handler in py_warn_logger.handlers:
            py_warn_logger.removeHandler(handler)
    log.handlers = []

    log_fmt = logging.Formatter(
        "%(asctime)s.%(msecs).3d - %(name)s - " "%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_fmt)
        log.addHandler(stream_handler)

    if file_output:
        file_handler = logging.FileHandler(file_output)
        file_handler.setFormatter(log_fmt)
        log.addHandler(file_handler)

    log.info("Logging started on AMICI version %s", amici.__version__)

    log.debug("OS Platform: %s", platform.platform())
    log.debug("Python version: %s", platform.python_version())
    log.debug("Hostname: %s", socket.getfqdn())

    if capture_warnings:
        logging.captureWarnings(capture_warnings)
        for handler in log.handlers:
            py_warn_logger.addHandler(handler)

    return log


def set_log_level(logger: logging.Logger, log_level: Union[int, bool]) -> None:
    if log_level is not None and log_level is not False:
        if isinstance(log_level, bool):
            log_level = logging.DEBUG
        elif not isinstance(log_level, int):
            raise ValueError("log_level must be a boolean, integer or None")

        if logger.getEffectiveLevel() != log_level:
            logger.debug(
                "Changing log_level from %d to %d"
                % (logger.getEffectiveLevel(), log_level)
            )
            logger.setLevel(log_level)


def get_logger(
    logger_name: Optional[str] = BASE_LOGGER_NAME,
    log_level: Optional[int] = None,
    **kwargs,
) -> logging.Logger:
    """
    Returns (if extistant) or creates an AMICI logger

    If the AMICI base logger has already been set up, this method will
    return it or any of its descendant loggers without overriding the
    settings - i.e. any values supplied as kwargs will be ignored.

    :param logger_name:
        Get a logger for a specific namespace, typically __name__
        for code outside of classes or self.__module__ inside a class

    :param log_level:
        Override the default or preset log level for the requested logger.
        None or False uses the default or preset value. True evaluates to
        logging.DEBUG. Any integer is used directly.

    :param console_output:
        Set up a default console log handler if True (default). Only used when
        the AMICI logger hasn't been set up yet.

    :param file_output:
        Supply a filename to copy all log output to that file, or set to
        False to disable (default). Only used when the AMICI logger hasn't
        been set up yet.

    :param capture_warnings:
        Capture warnings from Python's warnings module if True (default).
        Only used when the AMICI logger hasn't been set up yet..

    :return:
        A logging.Logger object with the requested name
    """
    if BASE_LOGGER_NAME not in logging.Logger.manager.loggerDict.keys():
        _setup_logger(**kwargs)
    elif kwargs:
        warnings.warn(
            "AMICI logger already exists, ignoring keyword "
            "arguments to setup_logger"
        )

    logger = logging.getLogger(logger_name)

    set_log_level(logger, log_level)

    return logger


def log_execution_time(description: str, logger: logging.Logger) -> Callable:
    """
    Parameterized function decorator that enables automatic execution time
    tracking

    :param description:
        Description of what the decorated function does

    :param logger:
        Logger to which execution timing will be printed
    """

    def decorator_timer(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            # append pluses to indicate recursion level
            recursion_level = sum(
                frame.function == "wrapper_timer"
                and frame.filename == __file__
                for frame in getouterframes(currentframe(), context=0)
            )

            recursion = ""
            level = logging.INFO
            level_length = len("INFO")
            if recursion_level > 1:
                recursion = "+" * (recursion_level - 1)
                level = logging.DEBUG
                level_length = len("DEBUG")

            tstart = time.perf_counter()
            rval = func(*args, **kwargs)
            tend = time.perf_counter()
            spacers = " " * max(
                59
                - len(description)
                - len(logger.name)
                - len(recursion)
                - level_length,
                0,
            )
            logger.log(
                level,
                f"Finished {description}{spacers}{recursion} ({(tend - tstart):.2E}s)",
            )
            return rval

        return wrapper_timer

    return decorator_timer
