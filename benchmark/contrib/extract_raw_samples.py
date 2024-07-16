import json

import pandas as pd


def _extract_raw_samples_from_row(row: pd.Series) -> pd.DataFrame:
    if pd.isna(row["raw_samples"]):
        return pd.DataFrame()
    raw_samples_df = pd.DataFrame(json.loads(row["raw_samples"]))
    # Merge with run configuration columns, dropping all aggregate stat cols for the run and the raw_samples col
    run_seconds_idx = row.index.tolist().index("run_seconds")
    util_95th_idx = row.index.tolist().index("util_95th")
    run_config_cols = (
        row.index.tolist()[:run_seconds_idx]
        + row.index.tolist()[util_95th_idx + 1 : -1]
    )
    _left_df = pd.concat(
        [row.to_frame().T[run_config_cols]] * len(raw_samples_df), ignore_index=True
    )
    # Rename context_tokens col before we merge
    _left_df.rename(columns={"context_tokens": "run_context_tokens"}, inplace=True)
    _left_df.index = raw_samples_df.index
    raw_samples_df = pd.merge(
        _left_df, raw_samples_df, left_index=True, right_index=True
    )
    return raw_samples_df


def _enrich_raw_samples_df(raw_samples_df: pd.DataFrame) -> pd.DataFrame:
    # Resource info
    raw_samples_df["platform_name"] = raw_samples_df["api_base_endpoint"].apply(
        lambda api_endpoint: (
            "openai" if "openai.com" in api_endpoint else "azure_openai"
        )
    )
    raw_samples_df["request_success"] = raw_samples_df.apply(
        lambda row: row["response_status_code"] == 200
        and row["last_exception"] is None
        and row["generated_tokens"] > 0,
        axis=1,
    )
    # Add latency cols
    raw_samples_df = raw_samples_df.copy()
    raw_samples_df["ttft_latency"] = raw_samples_df.apply(
        lambda row: (
            row["first_token_time"]
            - row["request_start_time"]
            - row["latency_adjustment_seconds"]
            if row["request_success"]
            else None
        ),
        axis=1,
    )
    raw_samples_df["e2e_latency"] = raw_samples_df.apply(
        lambda row: (
            row["response_end_time"]
            - row["request_start_time"]
            - row["latency_adjustment_seconds"]
            if row["request_success"]
            else None
        ),
        axis=1,
    )
    raw_samples_df["gen_latency"] = raw_samples_df.apply(
        lambda row: (
            row["response_end_time"] - row["first_token_time"]
            if row["request_success"]
            else None
        ),
        axis=1,
    )
    raw_samples_df["tbt_context"] = raw_samples_df.apply(
        lambda row: (
            row["ttft_latency"] / row["context_tokens"]
            if row["request_success"]
            else None
        ),
        axis=1,
    )
    raw_samples_df["tbt_gen"] = raw_samples_df.apply(
        lambda row: (
            row["gen_latency"] / row["generated_tokens"]
            if row["request_success"]
            else None
        ),
        axis=1,
    )
    return raw_samples_df


def get_extracted_raw_samples_df(
    combined_logs_df: pd.DataFrame, drop_failed_requests: bool = False
) -> pd.DataFrame:
    """
    Extracts all individual call data from the raw_samples column in a
    combined_logs Dataframe, returning a new Dataframe where each row is an
    individual request. Each row has its key statistics calculated based on the
    response start/end timestamps.

    Args:
        combined_logs_df: a combined_logs Dataframe.
        drop_failed_requests: If True, drops all requests that returned a
            non-200 status code, or where no tokens were generated. Defaults to
            False.

    Returns:
        A Dataframe of raw call data.
    """
    raw_samples_dfs = [
        _extract_raw_samples_from_row(row) for _, row in combined_logs_df.iterrows()
    ]
    raw_samples_df = pd.concat(
        raw_samples_dfs,
        ignore_index=True,
    )
    raw_samples_df = _enrich_raw_samples_df(raw_samples_df)
    if drop_failed_requests:
        raw_samples_df = raw_samples_df[raw_samples_df["request_success"]]
    return raw_samples_df
