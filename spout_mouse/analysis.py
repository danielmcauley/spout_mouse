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
    MOUSE_GROUPS
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
    lick_data_licks_per_spout["group"] = lick_data_licks_per_spout["mouse_id"].map(MOUSE_GROUPS)

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


def prepare_fp_dataframe(fp_df: pd.DataFrame, excluded_mice: list[str] = None) -> pd.DataFrame:
    """
    Prepare the fiber photometry DataFrame by converting mouse IDs to strings,
    mapping groups, and excluding specified mice.

    Parameters:
        fp_df (pd.DataFrame): The fiber photometry DataFrame.

    Returns:
        pd.DataFrame: The prepared DataFrame.
    """
    # Convert mouse_id from int to str
    fp_df["mouse_id"] = fp_df["mouse_id"].astype(str)

    # Map mouse_id to group
    fp_df["group"] = fp_df["mouse_id"].map(MOUSE_GROUPS)

    # Exclude certain mice
    if excluded_mice:
        fp_df = fp_df[~fp_df["mouse_id"].isin(excluded_mice)]

    return fp_df

def clean_fp_trials(fp_df: pd.DataFrame, num_trials: int) -> pd.DataFrame:
    """
    Ensure each mouse has a consistent number of trials and adjust trial numbers.

    Parameters:
        fp_df (pd.DataFrame): The fiber photometry DataFrame.
        num_trials (int): The number of trials to keep per mouse per day.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    fp_df_clean = (
        fp_df.sort_values(['cohort', 'day', 'mouse_id', 'trial_num'])
        .groupby(['cohort', 'day', 'mouse_id'])
        .tail(num_trials)
        .reset_index(drop=True)
    )
    fp_df_clean['trial_num'] = fp_df_clean.groupby(['cohort', 'day', 'mouse_id']).cumcount() + 1
    return fp_df_clean

def truncate_zscore_arrays(fp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Truncate the zscore_data_array in fp_df to the minimum length among all arrays.

    Parameters:
        fp_df (pd.DataFrame): The fiber photometry DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with truncated zscore_data_array.
    """
    min_length = fp_df["zscore_data_array"].apply(len).min()
    fp_df["zscore_data_array"] = fp_df["zscore_data_array"].apply(lambda x: x[:min_length])
    return fp_df

def merge_fp_with_lick_data(fp_df: pd.DataFrame, lick_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fiber photometry data with lick data to add spout information.

    Parameters:
        fp_df (pd.DataFrame): The fiber photometry DataFrame.
        lick_data (pd.DataFrame): The lick data DataFrame.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    unique_combinations = lick_data[["day", "cohort", "trial_num", "spout_name"]].drop_duplicates()
    fp_df_complete = pd.merge(fp_df, unique_combinations, on=['cohort', 'day', 'trial_num'], how='left')
    return fp_df_complete

def calculate_mean_zscore(fp_df: pd.DataFrame, across_days: bool = False) -> pd.DataFrame:
    """
    Calculate the mean zscore_data_array for each mouse, day, and spout_name.

    Parameters:
        fp_df (pd.DataFrame): The fiber photometry DataFrame.
        across_days (bool): Whether to calculate across days or per day.

    Returns:
        pd.DataFrame: DataFrame with mean zscore_data_array per mouse and spout_name.
    """
    if across_days:
        group_columns = ["mouse_id", "spout_name"]
    else:
        group_columns = ["mouse_id", "day", "spout_name"]
    mean_zscore_by_mouse_spout = (
        fp_df.groupby(group_columns)["zscore_data_array"]
        .apply(lambda x: np.mean(np.stack(x), axis=0))
        .reset_index()
    )

    # Map group information
    group_mapping = fp_df[["mouse_id", "group"]].drop_duplicates().set_index("mouse_id")["group"].to_dict()
    mean_zscore_by_mouse_spout["group"] = mean_zscore_by_mouse_spout["mouse_id"].map(group_mapping)

    return mean_zscore_by_mouse_spout

def prepare_long_format(mean_zscore_df: pd.DataFrame, across_days: bool = False) -> pd.DataFrame:
    """
    Transform the mean zscore DataFrame into long format for plotting.

    Parameters:
        mean_zscore_df (pd.DataFrame): DataFrame with mean zscore_data_array.
        across_days (bool): Whether data is calculated across days or per day.

    Returns:
        pd.DataFrame: Long format DataFrame for plotting.
    """
    # Explode zscore_data_array
    mean_zscore_long = mean_zscore_df.explode("zscore_data_array")

    # Add time index
    if across_days:
        groupby_columns = ["group", "mouse_id", "spout_name"]
    else:
        groupby_columns = ["group", "mouse_id", "day", "spout_name"]

    mean_zscore_long["time"] = mean_zscore_long.groupby(groupby_columns).cumcount()
    mean_zscore_long["zscore_data_array"] = mean_zscore_long["zscore_data_array"].astype(float)
    mean_zscore_long.rename(columns={'group':'group_name'}, inplace=True)

    # Convert day to string if needed
    if "day" in mean_zscore_long.columns:
        mean_zscore_long["day"] = mean_zscore_long["day"].astype(str)

    return mean_zscore_long

def calculate_mean_sem_zscores(mean_zscore_df: pd.DataFrame, across_days: bool = False) -> pd.DataFrame:
    """
    Calculate the mean and SEM of zscore_data_array for each group, day, and spout_name.

    Parameters:
        mean_zscore_df (pd.DataFrame): DataFrame with mean zscore_data_array.
        across_days (bool): Whether data is calculated across days or per day.

    Returns:
        pd.DataFrame: DataFrame with mean and SEM of zscore_data_array.
    """
    if across_days:
        group_columns = ["group", "spout_name"]
    else:
        group_columns = ["group", "day", "spout_name"]

    mean_sem_zscores = mean_zscore_df.groupby(group_columns)["zscore_data_array"].apply(
        lambda x: pd.DataFrame({
            'mean_zscore': np.mean(np.stack(x.values), axis=0),
            'sem_zscore': stats.sem(np.stack(x.values), axis=0, nan_policy='omit')
        })
    ).reset_index()

    # Explode the DataFrame to have one row per time point
    mean_sem_zscores = mean_sem_zscores.explode(['mean_zscore', 'sem_zscore']).reset_index(drop=True)

    # Add time index
    mean_sem_zscores['time'] = mean_sem_zscores.groupby(group_columns).cumcount()

    return mean_sem_zscores
