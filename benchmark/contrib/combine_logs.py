import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd


def combine_logs_to_csv(
    args: argparse.Namespace,
) -> None:
    """
    Combines all logs in a directory into a single csv file.

    Args:
        log_dir: Directory containing the log files.
        save_path: Path to save the output output CSV.
        load_recursive: Whether to load logs in all subdirectories of log_dir.
            Defaults to True.
    """
    if not args.save_path.endswith(".csv"):
        save_path = args.save_path + ".csv"
        logging.info(f"Warning: `save_path` does not end with .csv. Appending .csv to save_path. New path: {save_path}")
    log_dir = args.source_dir
    include_raw_request_info = args.include_raw_request_info
    stat_extraction_point = args.stat_extraction_point
    load_recursive = args.load_recursive

    log_dir = Path(log_dir)
    log_files = log_dir.rglob("*.log") if load_recursive else log_dir.glob("*.log")
    log_files = sorted(log_files)
    # Extract run info from each log file
    run_summaries = [extract_run_info_from_log_path(log_file, stat_extraction_point, include_raw_request_info) for log_file in log_files]
    run_summaries = [summary for summary in run_summaries if isinstance(summary, dict)]
    # Convert to dataframe and save to csv
    if run_summaries:
        df = pd.DataFrame(run_summaries)
        df.set_index("filename", inplace=True)
        df.to_csv(save_path, index=True)
        logging.info(f"Saved {len(df)} runs to {save_path}")
    else:
        logging.error(f"No valid runs found in {log_dir}")
    return


def extract_run_info_from_log_path(log_file: str, stat_extraction_point: str, include_raw_request_info: bool) -> Optional[dict]:
    """Extracts run info from log file path"""
    assert stat_extraction_point in ["draining", "final"], "stat_extraction_point must be either 'draining' or 'final'"
    is_format_human = False
    run_args = None
    last_logged_stats = None
    raw_samples = None
    early_terminated = False
    is_confirmed_as_ptu_endpoint = False
    is_draining_commenced = False
    prevent_reading_new_stats = False
    # Process lines, including only info BEFORE early termination (for terminated sessions), or the final log AFFTER requests start to drain (for valid sessions)
    with open(log_file) as f:
        for line in f.readlines():
            if "got terminate signal" in line:
                # Ignore any stats after early termination (since RPM, TPM, rate etc will start to decline as requests gradually finish)
                early_terminated = True
                break
            # Save most recent line
            if "Load" in line:
                run_args = json.loads(line.split("Load test args: ")[-1])
            if line.startswith("rpm:"):
                # Test was run with --output-format human. Cannot extract run args from this format.
                is_format_human = True
                break
            if "run_seconds" in line and not prevent_reading_new_stats:
                last_logged_stats = line
            if is_draining_commenced and stat_extraction_point == "draining":
                # Previous line was draining, use this line as the last set of valid stats
                prevent_reading_new_stats = True
            if "requests to drain" in line:
                # Current line is draining, next line is the last set of valid stats. Allow one more line to be processed.
                is_draining_commenced = True
            if include_raw_request_info and "Raw call stats: " in line:
                raw_samples = line.split("Raw call stats: ")[-1] # Do not load as json - output as string
    if is_format_human:
        logging.error(
            f"Could not extract run args from log file {log_file} - Data was collected with `--output-format human` (the default value). Please rerun the tests with `--output-format jsonl`."
        )
        return None
    if not run_args:
        logging.error(
            f"Could not extract run args from log file {log_file} - missing run info (it might have been generated with a previous code version)."
        )
        return None
    run_args["early_terminated"] = early_terminated
    run_args["filename"] = Path(log_file).name
    run_args["filepath"] = log_file
    # Extract last line of valid stats from log if available
    if last_logged_stats:
        last_logged_stats = flatten_dict(json.loads(last_logged_stats))
        run_args.update(last_logged_stats)
        run_args["run_has_non_throttled_failures"] = (
            int(run_args["failures"]) - int(run_args["throttled"]) > 0
        )
        is_confirmed_as_ptu_endpoint = last_logged_stats["util_avg"] != "n/a"
    run_args["is_confirmed_as_ptu_endpoint"] = is_confirmed_as_ptu_endpoint
    run_args["raw_samples"] = raw_samples
    return run_args


def flatten_dict(input: dict) -> dict:
    """
    Flattens dictionary of nested dictionaries/lists into a single level dictionary
    Taken from https://www.geeksforgeeks.org/flattening-json-objects-in-python/
    """
    out = {}

    def flatten(x, name=""):
        # If the Nested key-value
        # pair is of dict type
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + "_")

        # If the Nested key-value
        # pair is of list type
        elif isinstance(x, dict):
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            out[name[:-1]] = x

    flatten(input)
    return out


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="CLI for combining existing log files."
    )
    parser.add_argument(
        "source_dir", type=str, help="Directory containing the log files."
    )
    parser.add_argument(
        "save_path", type=str, help="Path to save the output output CSV."
    )
    parser.add_argument(
        "--include-raw-request-info", 
        type=str2bool, 
        nargs='?', 
        help="If True, all raw request info (timestamps, call status, request content) will be included for each individual request in every run where it is available.",
        const=True,
        default=True
    )
    parser.add_argument(
        "--stat-extraction-point", 
        type=str, 
        help="The point from which to extract statistics. If set to `draining`, stats are extraced when requests start draining, but before all requests have finished. If set to `final`, the very last line of stats are used, which could result in lower aggregate TPM/RPM numbers. See the README for more info.", 
        choices=["draining", "final"], 
        default="draining"
    )
    parser.add_argument(
        "--load-recursive",
        action="store_true",
        help="Whether to load logs in all subdirectories of log_dir.",
    )

    args = parser.parse_args()
    combine_logs_to_csv(args)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
main()
