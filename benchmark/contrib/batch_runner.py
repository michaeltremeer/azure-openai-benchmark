"""
This module can be used to run multiple runs of the benchmarking script with different permutations of parameters. 
Since this can be run at the command line, it also allows the running of testing across multiple deployments at the same time.

To use:
# Set the api key for the environment, e.g.
> export OPENAI_API_KEY=<your key>

# Run the tool for a single batch of runs (e.g. a cold-start warmup, followed by a combination of 2x workload-token-profiles and 2x concurrency values = 5x total runs)
> python -m benchmark.contrib.queue_runs --api-base-endpoint https://<YOUR_ENDPOINT>.openai.azure.com/ --deployment <MODEL_DEPLOYMENT> --log-save-dir logs --warmup-per-run 15 --cold-start-warmup 300 --aggregation-window 180 --concurrency-values 1,4 --workload-token-profiles 100-100,3000-500

# Run the tool for multiple batches of runs (e.g. 3x batches, with their start times 1 hour apart)
> python -m benchmark.contrib.queue_runs --api-base-endpoint https://<YOUR_ENDPOINT>.openai.azure.com/ --deployment <MODEL_DEPLOYMENT> --log-save-dir logs --warmup-per-run 15 --cold-start-warmup 300 --aggregation-window 180 --concurrency-values 1,4 --workload-token-profiles 100-100,3000-500 --num-batches 3 --batch-repeat-delay 3600

# Combine the logs with the combine_logs tool
> python -m benchmark.contrib.combine_logs logs logs/combined_runs.csv --load-recursive
"""

import argparse
import json
import os
import shlex
import subprocess
import time
from typing import Iterable, Optional, Union

from requests import post

from ..oairequester import TELEMETRY_USER_AGENT_HEADER, USER_AGENT, UTILIZATION_HEADER


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Create argparse parser for run_configs
def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-workload benchmarking.")
    parser.add_argument(
        "api_base_endpoint", help="Azure OpenAI deployment base endpoint.", nargs=1
    )
    parser.add_argument(
        "--deployment", type=str, help="Azure OpenAI deployment name.", required=True
    )
    parser.add_argument(
        "--context-generation-method",
        type=str,
        default="generate",
        help="Context generation method - determines whether to generate the context tokens or replay messages from a file.",
        choices=["generate", "replay"],
    )
    parser.add_argument(
        "--token-rate-workload-list",
        type=str,
        default="none",
        help="Comma-separated list of all workload args to test, in the order of <context-tokens>-<max-tokens>-<rate>. e.g. '500-100-20,3500-300-none' when context-generation-method=generate, or 'replay_messages_1.json-100-10,replay_messages_2.json-200-20' when context-generation-method=replay",
        required=True,
    )
    parser.add_argument(
        "--aggregation-window",
        type=int,
        default=120,
        help="Length of time to collect and aggregate statistcs for each run. Defaults to 120.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Max Duration to run each benchmark run.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        help="Minimum number of requests to include in each benchmark run.",
    )
    parser.add_argument(
        "--run-end-condition-mode",
        type=str,
        help="Determines whether both the `requests` and `duration` args must be reached before ending the run ('and'), or whether to end the run either either arg is reached ('or'). Defaults to 'or'.",
        choices=["and", "or"],
    )
    parser.add_argument(
        "--clients",
        type=int,
        default="20",
        help="Number of clients to use for each run. Defaults to 20.",
    )
    parser.add_argument(
        "--start-ptum-runs-at-full-utilization",
        type=str2bool,
        nargs="?",
        help="Starts all PTU-M runs at 100% utilization, preventing any burst capacity from inflating the results. Defaults to True.",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--log-save-dir",
        type=str,
        help="If provided, will save stddout to this directory. Filename will include important run parameters.",
    )
    parser.add_argument(
        "--log-request-content",
        type=str2bool,
        nargs="?",
        help="If True, will log the raw input and output tokens of every request. Defaults to False.",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--retry",
        type=str,
        default="none",
        help="Request retry strategy.",
        choices=["none", "exponential"],
    )
    parser.add_argument(
        "--frequency-penalty", type=float, help="Request frequency_penalty."
    )
    parser.add_argument(
        "--presence-penalty", type=float, help="Request frequency_penalty."
    )
    parser.add_argument("--temperature", type=float, help="Request temperature.")
    parser.add_argument("--top-p", type=float, help="Request top_p.")
    parser.add_argument(
        "--prevent-server-caching",
        type=str2bool,
        nargs="?",
        help="Adds a random prefixes to all requests in order to prevent server-side caching. Defaults to True.",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable that contains the API KEY.",
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default="2023-05-15",
        help="Set OpenAI API version.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of times to repeat the full batch of benchmarks (including cold-start-warmup). Defaults to 1 (a single batch).",
    )
    parser.add_argument(
        "--batch-start-interval",
        type=int,
        default=3600,
        help="Seconds to wait between the start of each batch of runs (NOT from the end of one to the start of the next). Defaults to 3600 seconds (1 hour).",
    )
    return parser.parse_args()


def benchmark_args_to_exec_str(
    api_base_endpoint: str,
    deployment: str,
    context_generation_method: str,
    max_tokens: int,
    aggregation_window: int,
    clients: int,
    prevent_server_caching: bool,
    retry: str,
    context_tokens: Optional[int] = None,
    replay_path: Optional[str] = None,
    rate: Optional[float] = None,
    duration: Optional[int] = None,
    requests: Optional[int] = None,
    run_end_condition_mode: Optional[str] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    log_save_dir: Optional[str] = None,
    log_request_content: Optional[bool] = None,
    api_key_env: str = "OPENAI_API_KEY",
):
    """Converts args into an execution string for the benchmarking script."""
    if context_generation_method == "generate":
        context_source_str = f"--context-tokens {context_tokens}"
    else:
        context_source_str = f"--replay-path {replay_path}"
    # Add required parameters
    cmd = (
        f"python3 -m benchmark.bench load {api_base_endpoint} --deployment {deployment} {context_source_str}"
        f" --max-tokens {max_tokens} --output-format jsonl --aggregation-window {aggregation_window} --clients {clients} "
        f"--prevent-server-caching {prevent_server_caching} --retry {retry} --api-key-env {api_key_env} "
        f"--context-generation-method {context_generation_method} --shape custom"
    )
    # Add optionals
    if rate is not None:
        cmd += f" --rate {rate}"
    if duration is not None:
        cmd += f" --duration {duration}"
    if requests is not None:
        cmd += f" --requests {requests}"
    if run_end_condition_mode is not None:
        cmd += f" --run-end-condition-mode {run_end_condition_mode}"
    if log_save_dir is not None:
        cmd += f" --log-save-dir {log_save_dir}"
    if log_request_content is not None:
        cmd += f" --log-request-content {log_request_content}"
    if frequency_penalty is not None:
        cmd += f" --frequency-penalty {requests}"
    if presence_penalty is not None:
        cmd += f" --presence-penalty {requests}"
    if temperature is not None:
        cmd += f" --temperature {requests}"
    if top_p is not None:
        cmd += f" --top-p {requests}"
    return cmd


def run_benchmark_exec_str(
    exec_str: str,
    print_terminal_output: bool = True,
    kill_when_draining_begins: bool = True,
    kill_at_100_util: bool = False,
) -> None:
    """
    Runs a benchmark execution string, optionally killing the run if certain criteria are met.
    :param print_terminal_output: If True, the terminal output will be printed to the console.
    :param exec_str: Terminal command to be executed.
    :param kill_when_draining_begins: If True, the run will be killed as soon as requests start to drain. This prevents PTU utilization dropping as the last requests finish.
    :param kill_at_100_util: If True and the endpoint is a PTU-M model deployment, the run will be killed as soon as utilization 95th is above 98%. This ensures the endpoint has no 'burst credits' prior to the next run.
    """
    # try:
    process = subprocess.Popen(
        shlex.split(exec_str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    draining_started = False

    while True:
        nextline = process.stdout.readline().decode("unicode_escape")
        if nextline == "" and process.poll() is not None:
            break

        if nextline:
            if print_terminal_output:
                print(nextline.strip())
            # Kill process if utilization exceeds 98%
            if kill_at_100_util and '"util":' in nextline:
                # Load utilization - should be last subdict in the output - should be one of either:
                # PayGO or no responses received yet: "{..., "util": {"avg": "n/a", "95th": "n/a"}}"
                # PTU and first response is received: "{..., "util": {"avg": "74.2%", "95th": "78.5%"}}"
                util_dict = json.loads(nextline.split('"util": ')[1][:-2])
                last_util_95th = util_dict["95th"]
                if last_util_95th != "n/a":
                    last_util_95th = float(last_util_95th[:-2])
                    if last_util_95th > 98:
                        print(
                            "PTU-M utilization exceeded 98% - terminating warmup run process"
                        )
                        process.kill()
            # Kill process if run draining has occurred. Make sure to kill process after one more line of stats has been logged.
            if kill_when_draining_begins and draining_started:
                print(
                    "Draining detected and final stats are logged - terminating process immediately."
                )
                process.kill()
            if kill_when_draining_begins:
                # Set drain var so run is killed after next line is processed
                if "drain" in nextline:
                    draining_started = True
    return


def run_benchmark_batch(
    api_base_endpoint: str,
    deployment: str,
    context_generation_method: str,
    token_rate_workload_list: Iterable[tuple[Union[str, int], int, Union[None, float]]],
    aggregation_window: int,
    duration: Optional[int],
    requests: Optional[int],
    run_end_condition_mode: str,
    clients: Optional[int],
    log_save_dir: str,
    log_request_content: Optional[bool],
    prevent_server_caching: bool,
    start_ptum_runs_at_full_utilization: bool,
    retry: str,
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    temperature: Optional[float],
    top_p: Optional[float],
    api_key_env: str,
    api_version: str,
) -> None:
    """
    Runs a batch benchmarks for all token/rate combos.
    :param api_base_endpoint: Azure OpenAI deployment base endpoint.
    :param deployment: Azure OpenAI deployment name.
    :param context_generation_method: Context generation method - determines whether to generate the context tokens or replay messages from a file.
    :param token_rate_workload_list: List of (context_tokens OR replay_path, max_tokens, rate) tuples.
    :param aggregation_window: Period of time over which to aggregate run statistcs.
    :param duration: Duration of each run.
    :param requests: Max number of requests in each run.
    :param run_end_condition_mode: Determines whether both the `requests` and `duration` args must be reached before ending the run ('and'), or whether to end the run either either arg is reached ('or'). Defaults to 'or'.
    :param clients: Number of clients to use in each test.
    :param log_save_dir: Will save all logs to this directory.
    :param log_request_content: If True, will log the raw input and output content of every request.
    :param prevent_server_caching: Whether to prevent server caching in each test.
    :param start_ptum_runs_at_full_utilization: For PTU-M deployments, run a high load run through the endpoint prior to each and every benchmark run to ensure benchmnark runs start at 100% utilization (avoiding the effect of burst capacity influencing the results).
    :param retry: Request retry strategy.
    :param frequency_penalty: Request frequency_penalty.
    :param presence_penalty: Request presence_penalty.
    :param temperature: Request temperature.
    :param top_p: Request top_p.
    :param api_key_env: Environment variable that contains the API KEY.
    :param api_version: API version to use. Defaults to '2023-05-15'.
    """
    is_ptu_deployment = None
    if start_ptum_runs_at_full_utilization:
        print("Checking whether endpoint is PTU-M deployment...")
        # Send a test request to check whether the endpoint returns a utilization header
        api_key = os.getenv(api_key_env)
        url = (
            api_base_endpoint
            + "/openai/deployments/"
            + deployment
            + "/chat/completions"
        )
        url += "?api-version=" + api_version
        util_check_headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
            TELEMETRY_USER_AGENT_HEADER: USER_AGENT,
        }
        util_check_body = {
            "messages": [{"content": "What is 1+1?", "role": "user"}],
        }
        response = post(url, headers=util_check_headers, json=util_check_body)
        if response.status_code != 200:
            raise ValueError(
                f"Deployment type check failed with code {response.status_code}. Reason: {response.reason}. Data: {response.text}"
            )
        if UTILIZATION_HEADER in response.headers:
            print(
                "Utilization header found in endpoint response. This is a PTU-M deployment and will be warmed up prior to each benchmark run."
            )
            is_ptu_deployment = True
        else:
            print(
                "Utilization header not found in endpoint response. This is not a PTU-M deployment - no endpoint warmup is necessary."
            )
            is_ptu_deployment = False

    # Run the actual tests
    for run_num, (context_input_arg, max_tokens, rate) in enumerate(
        token_rate_workload_list
    ):
        if start_ptum_runs_at_full_utilization and is_ptu_deployment:
            print(
                "Running high load through PTU-M endpoint to push utilization to 100%..."
            )
            # Run high load until the PTU-M deployment is at 100% util, then kill the run
            ptu_exec_str = benchmark_args_to_exec_str(
                api_base_endpoint=api_base_endpoint,
                deployment=deployment,
                context_generation_method=context_generation_method,
                context_tokens=500,
                max_tokens=100,
                rate=None,
                log_save_dir=None,
                log_request_content=False,
                aggregation_window=60,
                duration=None,
                requests=None,
                clients=20,
                prevent_server_caching=True,
                retry="none",
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                top_p=top_p,
                api_key_env=api_key_env,
            )
            run_benchmark_exec_str(
                exec_str=ptu_exec_str,
                print_terminal_output=False,
                kill_when_draining_begins=True,
                kill_at_100_util=True,
            )
        # Run actual benchmark run, killing after request draining (to avoid wasting time or letting utilization drop between runs)
        if context_generation_method == "generate":
            context_tokens = context_input_arg
            replay_path = None
        else:
            context_tokens = None
            replay_path = context_input_arg
        print(f"Starting benchmark {run_num+1} of {len(token_rate_workload_list)}")
        benchmark_exec_str = benchmark_args_to_exec_str(
            api_base_endpoint=api_base_endpoint,
            deployment=deployment,
            context_generation_method=context_generation_method,
            context_tokens=context_tokens,
            replay_path=replay_path,
            max_tokens=max_tokens,
            rate=rate,
            log_save_dir=log_save_dir,
            log_request_content=log_request_content,
            aggregation_window=aggregation_window,
            duration=duration,
            requests=requests,
            run_end_condition_mode=run_end_condition_mode,
            clients=clients,
            prevent_server_caching=prevent_server_caching,
            retry=retry,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            api_key_env=api_key_env,
        )
        run_benchmark_exec_str(
            exec_str=benchmark_exec_str,
            print_terminal_output=True,
            kill_when_draining_begins=False,
            kill_at_100_util=False,
        )


def validate_and_process_context_token_workload_list(
    token_rate_workload_list: str, context_generation_method: str
) -> list:
    """Checks the format and content of token_rate_workload_list argument."""
    valid_context_generation_methods = ("generate", "replay")
    if context_generation_method not in valid_context_generation_methods:
        raise ValueError(
            f"context-generation-method invalid - must be one of {valid_context_generation_methods}"
        )
    if " " in token_rate_workload_list:
        raise ValueError("Error: token-rate-workload-list must not contain spaces.")
    output = list()
    for item in token_rate_workload_list.split(","):
        split_vals = item.split("-")
        if not len(split_vals) == 3:
            if context_generation_method == "generate":
                exc_string = f"Invalid token-rate-workload item '{item}'. Expected format: <context-tokens>-<max-tokens>-<rate> - e.g. '500-100-8.5'."
            else:
                exc_string = f"Invalid token-rate-workload item '{item}'. Expected format: <replay-filepath>-<max-tokens>-<rate> - e.g. 'replay_messages.json-100-10'. Ensure there are no dashes in the filename"
            raise ValueError(exc_string)
        if context_generation_method == "generate":
            try:
                context_definition = int(split_vals[0])
            except Exception as e:
                raise ValueError(
                    f"When context-generation-method = generate, the first value in each token-rate-workload item must be a valid integer. '{split_vals[0]}' is not a valid integer."
                )
        else:
            context_definition = split_vals[0]
            if not os.path.exists(context_definition):
                raise ValueError(
                    f"Replay filepath '{context_definition}' not found. Make sure the first value in each token-rate-workload item is a valid filepath (relative to the directory from which the command is being run)."
                )
        max_tokens = int(split_vals[1])
        if split_vals[2].lower() == "none":
            rate = None
        else:
            rate = float(split_vals[2])
        output.append((context_definition, max_tokens, rate))
    return output


def main():
    args = parse_args()
    # Parse workload-token-profiles
    token_rate_workload_list = validate_and_process_context_token_workload_list(
        args.token_rate_workload_list, args.context_generation_method
    )
    api_base_endpoint = args.api_base_endpoint[0]

    try:
        if args.num_batches == 1:
            log_str = "Running one batch of the following workloads:"
            context_source_logging_str = (
                "context_tokens"
                if args.context_generation_method == "generate"
                else "replay_filepath"
            )
            for run_num, token_rate_workload in enumerate(
                token_rate_workload_list, start=1
            ):
                log_str += f"\n - {run_num}. {context_source_logging_str}: {token_rate_workload[0]}, max_tokens: {token_rate_workload[1]}, rate: {token_rate_workload[2]}"
            print(log_str)
            start_time = time.time()
            # Single-batch runs
            run_benchmark_batch(
                api_base_endpoint=api_base_endpoint,
                deployment=args.deployment,
                context_generation_method=args.context_generation_method,
                token_rate_workload_list=token_rate_workload_list,
                aggregation_window=args.aggregation_window,
                duration=args.duration,
                requests=args.requests,
                run_end_condition_mode=args.run_end_condition_mode,
                clients=args.clients,
                log_save_dir=args.log_save_dir,
                log_request_content=args.log_request_content,
                prevent_server_caching=args.prevent_server_caching,
                start_ptum_runs_at_full_utilization=args.start_ptum_runs_at_full_utilization,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                temperature=args.temperature,
                top_p=args.top_p,
                retry=args.retry,
                api_key_env=args.api_key_env,
                api_version=args.api_version,
            )
            print(f"Batch complete in {int(time.time() - start_time)} seconds.")
        else:
            # Multi-batch runs
            # Sanity check batch repeat amount based on duration per run
            if args.duration:
                expected_time_per_batch = sum(
                    [len(token_rate_workload_list) * args.duration + 15]
                )
                if expected_time_per_batch > args.batch_start_interval:
                    print(
                        f"WARNING: Batch repeat delay ({args.batch_start_interval}s) is less than the expected time per batch ({expected_time_per_batch}s). This may result in overlapping runs."
                    )
            start_time = time.time()
            runs_completed = 0
            while runs_completed < args.num_batches:
                print(f"Starting batch {runs_completed+1} of {args.num_batches}")
                run_benchmark_batch(
                    api_base_endpoint=api_base_endpoint,
                    deployment=args.deployment,
                    context_generation_method=args.context_generation_method,
                    token_rate_workload_list=token_rate_workload_list,
                    aggregation_window=args.aggregation_window,
                    duration=args.duration,
                    requests=args.requests,
                    run_end_condition_mode=args.run_end_condition_mode,
                    clients=args.clients,
                    log_save_dir=args.log_save_dir,
                    log_request_content=args.log_request_content,
                    prevent_server_caching=args.prevent_server_caching,
                    start_ptum_runs_at_full_utilization=args.start_ptum_runs_at_full_utilization,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    retry=args.retry,
                    api_key_env=args.api_key_env,
                    api_version=args.api_version,
                )
                runs_completed += 1
                if runs_completed < args.num_batches:
                    secs_to_wait = int(
                        (start_time + args.batch_start_interval * runs_completed)
                        - time.time()
                    )
                    if secs_to_wait > 0:
                        print(
                            f"Batch complete. Waiting {secs_to_wait} seconds before starting next batch..."
                        )
                        time.sleep(secs_to_wait)
                    else:
                        print(
                            f"WARNING: Batch {runs_completed+1} took longer than {args.batch_start_interval} seconds. Starting next batch immediately."
                        )
            print("All batches complete.")
        return
    except KeyboardInterrupt as _kbi:
        print("keyboard interrupt detected. exiting...")
        return
    except Exception as e:
        raise e


main()
