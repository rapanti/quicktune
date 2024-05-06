from datetime import datetime
from pathlib import Path
import logging
import os
from time import sleep
from random import randint

logger = logging.getLogger(__name__)


def setup_outputdir(path, warn_if_exist=True, create_dir=True):
    if path:
        assert isinstance(
            path, (str, Path)
        ), f"Only str and pathlib.Path types are supported for path, got {path} of type {type(path)}."

    if path is None:
        for _ in range(1000):
            try:
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                path = os.path.join("QuickTuner", f"qtt-{timestamp}")
                os.makedirs(path, exist_ok=False)
                break
            except FileExistsError:
                sleep(randint(1, 10))
                continue
        else:
            raise RuntimeError("Too many jobs startet at the same time.")
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
    elif warn_if_exist:
        try:
            if create_dir:
                os.makedirs(path, exist_ok=False)
            elif os.path.isdir(path):
                raise FileExistsError
        except FileExistsError:
            logger.warning(
                f'Warning: path already exists! This may overwrite previous runs! path="{path}"'
            )
    path = os.path.expanduser(path)  # replace ~ with absolute path if it exists
    return path
