import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import auc
from typing import List
from .config import (
    TRIAL_START,
    TRIAL_END,
    BASELINE_START,
    BASELINE_END,
    MOUSE_GROUPS,
    NUM_TRIALS
)
from .signal_processing import detrend_signal
from .data_loading import build_traces_df, build_spout_df


def calculate_total_licks_per_trial(lick_data_complete: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the total number of licks per trial for each mouse.

    Args:
        lick_data_complete (pd.DataFrame): DataFrame containing complete lick data.

    Returns:
        pd.DataFrame: DataFrame with columns 'mouse_id', 'day', 'trial_num', 'lick_count_total',
            'spout_name', and 'group'.
    """
    lick_data_each_trial_total_licks = (
        lick_data_complete.groupby(["mouse_id", "day", "trial_num"])
        .agg({"lick_count": "sum", "spout_name": "first", "group": "first"})
        .reset_index()
        .rename(columns={"lick_count": "lick_count_total"})
    )

    return lick_data_each_trial_total_licks


def calculate_average_licks_per_spout(lick_data_each_trial_total_licks: pd.DataFrame, combine_days: bool = True) -> pd.DataFrame:
    """
    Calculates the average number of licks per spout across trials and mice.

    Args:
        lick_data_each_trial_total_licks (pd.DataFrame): DataFrame with total lick counts per trial.

    Returns:
        pd.DataFrame: DataFrame with columns 'mouse_id', 'day', 'spout_name', 'lick_count_total', and 'group'.
    """
    groups = ["mouse_id", "spout_name"]
    if not combine_days:
        groups += ["day"]

    lick_data_licks_per_spout = (
        lick_data_each_trial_total_licks.groupby(groups)
        .agg({"lick_count_total": "mean"})
        .reset_index()
    )
    lick_data_licks_per_spout["group"] = lick_data_licks_per_spout["mouse_id"].map(config.MOUSE_GROUPS)

    return lick_data_licks_per_spout


def organize_lick_data_by_spout(lick_data_complete: pd.DataFrame) -> pd.DataFrame:
    """
    Organizes lick data by sorting and grouping it by 'mouse_id', 'group', 'day', 'spout_name', and 'time_ms_binned',
    and calculating the average lick count per second ('lick_count_hz').

    Args:
        lick_data_complete (pd.DataFrame): DataFrame containing complete lick data.

    Returns:
        pd.DataFrame: DataFrame with columns 'mouse_id', 'group', 'day', 'spout_name', 'time_ms_binned', and 'lick_count_avg'.
    """
    lick_data_spout = lick_data_complete \
        .sort_values(["mouse_id", "group", "day", "spout_name", "time_ms_binned"]) \
        .groupby(["mouse_id", "group", "day", "spout_name", "time_ms_binned"]) \
        .agg(lick_count_avg=("lick_count_hz", "mean"))

    return lick_data_spout.reset_index()


def aggregate_data_and_calculate_sem(lick_data_spout: pd.DataFrame, combine_days: bool = True) -> pd.DataFrame:
    """
    Aggregates lick data by grouping it by 'group', 'spout_name', and 'time_ms_binned',
    and calculates the mean lick count per second ('lick_count_avg') and SEM.

    Args:
        lick_data_spout (pd.DataFrame): DataFrame containing organized lick data.
        sem_func (function): Function to calculate SEM.

    Returns:
        pd.DataFrame: DataFrame with aggregated data, including columns 'group', 'spout_name', 'time_ms_binned',
            'lick_avg_all', and 'sem'.
    """
    groups = ["group", "spout_name", "time_ms_binned"]
    if not combine_days:
        groups += ["day"]

    def sem_func(arr):
        return stats.sem(arr, axis=None, ddof=0)

    lick_data_grouped = lick_data_spout.groupby(groups).agg(
        lick_avg_all=("lick_count_avg", "mean"),
        sem=("lick_count_avg", sem_func)
    ).reset_index()

    return lick_data_grouped


def calculate_zscores(
    spout_data: pd.DataFrame,
    gcamp_data: pd.DataFrame,
    trial_start: int = TRIAL_START,
    trial_end: int = TRIAL_END,
    baseline_start: int = BASELINE_START,
    baseline_end: int = BASELINE_END
) -> pd.DataFrame:
    """
    Calculate z-scores for each trial based on baseline data.

    Parameters:
        spout_data (pd.DataFrame): DataFrame containing spout extension timestamps.
        gcamp_data (pd.DataFrame): DataFrame containing detrended GCaMP data.
        trial_start (int): Start time relative to spout extension for trial.
        trial_end (int): End time relative to spout extension for trial.
        baseline_start (int): Start time relative to spout extension for baseline.
        baseline_end (int): End time relative to spout extension for baseline.

    Returns:
        pd.DataFrame: DataFrame with z-score data arrays added.
    """
    zscore_data_array_list = []

    for _, row in spout_data.iterrows():
        spout_ext_sec = row["spout_extension_timestamp_sec"]

        baseline_data = gcamp_data.query(
            f'{spout_ext_sec} - {baseline_start} <= timestamp_sec <= {spout_ext_sec} - {baseline_end}'
        )
        zscore_data = gcamp_data.query(
            f'{spout_ext_sec} - {trial_start} <= timestamp_sec <= {spout_ext_sec} + {trial_end}'
        )

        baseline_mean = baseline_data["detrended_gcamp"].mean()
        baseline_sd = baseline_data["detrended_gcamp"].std()

        z_scores = (zscore_data["detrended_gcamp"] - baseline_mean) / baseline_sd
        zscore_data_array_list.append(z_scores.values)

    spout_data["zscore_data_array"] = zscore_data_array_list

    return spout_data


def add_auc(data: pd.DataFrame, sample_rate: float, downsample_rate: int) -> pd.DataFrame:
    """
    Add area under the curve (AUC) to the data.

    Parameters:
        data (pd.DataFrame): DataFrame containing z-score data arrays.
        sample_rate (float): Original sample rate.
        downsample_rate (int): Downsample rate.

    Returns:
        pd.DataFrame: DataFrame with 'auc' column added.
    """
    def get_subset_auc(zscore_array: np.ndarray, sample_rate: float = sample_rate, start_sec: int = 0, end_sec: int = 5) -> float:
        total_time = (downsample_rate / sample_rate) * len(zscore_array)
        time_vector = np.linspace(0, total_time, len(zscore_array))
        mask = (time_vector >= start_sec) & (time_vector <= end_sec)
        y = zscore_array[mask]
        x = time_vector[mask]
        return auc(x, y)

    data["auc"] = data["zscore_data_array"].apply(get_subset_auc)
    return data

