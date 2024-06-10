import abc
import datetime
import logging
import os
import sys


import yaml
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler


class PipelineRunner(abc.ABC):
    @classmethod
    def run(cls, task_fn: str):
        # Setup logging
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        stdout_handler = RichHandler(highlighter=NullHighlighter())
        stdout_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(stdout_handler)

        sys.path.insert(0, os.path.realpath(os.curdir))

        os.chdir(os.path.dirname(task_fn) or ".")

        task_name = os.path.splitext(os.path.basename(task_fn))[0]
        task_fn_modified = datetime.datetime.fromtimestamp(os.stat(task_fn).st_mtime)

        log_fn = os.path.abspath(
            f'{task_name}-{datetime.datetime.now().isoformat(timespec="seconds")}.log'
        )
        print(f"Logging to {log_fn}.")
        file_handler = logging.FileHandler(log_fn)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

        # Also capture exceptions

        def log_except_hook(*exc_info):
            root_logger.error("Unhandled exception", exc_info=exc_info)  # type: ignore

        sys.excepthook = log_except_hook

        # # Also capture py.warnings
        # warnings_logger = logging.getLogger("py.warnings")
        # warnings_logger.addHandler(file_handler)
        # warnings_logger.addHandler(stream_handler)
        # warnings_logger.setLevel(logging.INFO)

        root_logger.info(
            f"Loading pipeline config from {task_fn} ({task_fn_modified.isoformat(timespec='seconds')})"
        )

        # logging.getLogger("zoomie2").setLevel(logging.DEBUG)

        log_levels = {
            name: logging.getLevelName(logging.getLogger(name).getEffectiveLevel())
            for name in sorted(root_logger.manager.loggerDict)
        }
        root_logger.info(f"Log levels: {log_levels}")

        with open(task_fn) as f:
            config_dict = yaml.safe_load(f)

        cls._configure_and_run(config_dict)

        root_logger.info("Finished processing.")

    @staticmethod
    @abc.abstractmethod
    def _configure_and_run(config_dict): ...
