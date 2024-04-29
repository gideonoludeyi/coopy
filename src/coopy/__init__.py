import argparse
import logging
import sys

import coopy.experiment as experiment
import coopy.run as run
import coopy.execute as execute

parser = argparse.ArgumentParser("coopy")
subparser_creator = parser.add_subparsers(required=True)

run.setup_parser(subparser_creator.add_parser("run"))
experiment.setup_parser(subparser_creator.add_parser("experiment"))
execute.setup_parser(subparser_creator.add_parser("execute"))

# Configure logging to display info level logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> int:
    args = parser.parse_args()
    args.__exec__(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
