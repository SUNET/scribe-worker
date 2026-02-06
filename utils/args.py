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
        "--mode",
        type=str,
        choices=["all", "inference", "transcription"],
        default="all",
        help="Run mode: 'all' (default), 'inference' only, or 'transcription' only.",
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
        args.mode,
    )
