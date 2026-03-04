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

import argparse


def parse_arguments() -> tuple:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Transcription worker")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground mode.",
    )
    parser.add_argument(
        "--pidfile",
        type=str,
        default="/tmp/worker.pid",
        help="Path to PID file.",
    )
    parser.add_argument(
        "--zap",
        action="store_true",
        help="Zap the existing PID file.",
    )
    parser.add_argument(
        "--envfile",
        type=str,
        default=".env",
        help="Path to environment file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )

    parser.add_argument(
        "--logfile",
        type=str,
        default="",
        help="Path to log file.",
    )

    parser.add_argument(
        "--no-healthcheck",
        action="store_true",
        help="Disable healthcheck thread.",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all configured models and exit.",
    )

    args = parser.parse_args()

    return (
        args.foreground,
        args.pidfile,
        args.zap,
        args.envfile,
        args.debug,
        args.logfile,
        args.no_healthcheck,
        args.download,
    )
