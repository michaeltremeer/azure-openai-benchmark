import argparse
import logging

from ..jsonloganalysis import combine_logs_to_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="CLI for combining existing log files.")
    parser.add_argument("source_dir", type=str, help="Directory containing the log files.")
    parser.add_argument("save_path", type=str, help="Path to save the output output CSV.")
    parser.add_argument("--load-recursive", action="store_true", help="Whether to load logs in all subdirectories of log_dir.")

    args = parser.parse_args()
    combine_logs_to_csv(args)

main()