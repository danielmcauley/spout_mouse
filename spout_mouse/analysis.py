import pandas as pd
import numpy as np
from scipy import stats
from spout_mouse import config
import itertools


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
