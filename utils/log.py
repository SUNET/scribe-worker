# Copyright (c) 2025-2026 Sunet.
# Contributor: Kristofer Hallin
#
# This file is part of Sunet Scribe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from utils.args import parse_arguments


def get_logger():
    """
    Get a logger instance for the application. If the logger already has
    handlers, it returns the existing logger.
    """

    logger = logging.getLogger(__name__)
    _, _, _, _, debug, logfile, _ = parse_arguments()

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
