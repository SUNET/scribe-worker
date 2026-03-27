import argparse


def parse_arguments() -> tuple:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Transcription worker")
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
        "--download",
        action="store_true",
        help="Download all configured models and exit.",
    )

    args = parser.parse_args()

    return (
        args.envfile,
        args.debug,
        args.logfile,
        args.download,
    )
