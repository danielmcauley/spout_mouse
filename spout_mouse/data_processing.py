import os
import glob
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
from .config import (
    LICK_CODES,
    SPOUT_EXT_CODE,
    SPOUT_POS_CODE,
    VALID_SPOUT_POS,
    LICK_DATA_COLS,
    BIN_SIZE_MS,
    MOUSE_GROUPS,
)


def extract_zip_files(zip_file_paths: List[str], extract_to: str) -> None:
    """
    Extract multiple zip files to a specified directory.

    Parameters:
        zip_file_paths (List[str]): List of paths to zip files.
        extract_to (str): Directory to extract files into.

    Raises:
        FileNotFoundError: If any zip file is not found.
        zipfile.BadZipFile: If any zip file is corrupted.
    """
    os.makedirs(extract_to, exist_ok=True)
    for zip_path in zip_file_paths:
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)


def load_experiment_metadata(
    experiment_df: pd.DataFrame,
    experiment_name: str,
    cohorts: List[int],
    days: List[int]
) -> pd.DataFrame:
    """
    Filter and load experiment metadata from DataFrame.

    Parameters:
        experiment_df (pd.DataFrame): DataFrame containing experiment data.
        experiment_name (str): Name of the experiment to filter.
        cohorts (List[int]): List of cohort numbers to include.
        days (List[int]): List of days to include.

    Returns:
        pd.DataFrame: Filtered DataFrame with spout information.

    Raises:
        ValueError: If required columns are missing in experiment_df.
    """
    required_columns = {'experiment', 'cohort', 'day', 'spout_id', 'spout_name'}
    if not required_columns.issubset(experiment_df.columns):
        missing = required_columns - set(experiment_df.columns)
        raise ValueError(f"Missing required columns in experiment_df: {missing}")
    
    filtered_df = experiment_df[
        (experiment_df['experiment'] == experiment_name) &
        (experiment_df['cohort'].isin(cohorts)) &
        (experiment_df['day'].isin(days))
    ]
    spout_names = filtered_df[['cohort', 'day', 'spout_id', 'spout_name']].drop_duplicates().reset_index(drop=True)
    return spout_names


def process_lick_data(data_directory: str) -> pd.DataFrame:
    """
    Process lick data from CSV files into a structured DataFrame.

    Parameters:
        data_directory (str): Directory containing CSV data files.

    Returns:
        pd.DataFrame: Processed lick data.

    Raises:
        FileNotFoundError: If no CSV files are found in the directory.
    """

    all_files = glob.glob(os.path.join(data_directory, '**', '*.csv'), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_directory}")

    lick_data_list = []
    for file_path in tqdm(all_files, desc="Processing Lick Data"):

        lick_data = pd.read_csv(file_path, usecols=[0], skiprows=1, names=["event_pos_time"]).dropna()
        lick_data[["event_tag", "time_ms"]] = lick_data["event_pos_time"].str.split(expand=True)
        lick_data.drop(["event_pos_time"], axis=1, inplace=True)

        # Forward fill the 'spout_id' for events where 'event_tag' equals the spout position code
        lick_data["spout_id"] = lick_data["time_ms"].astype(int) \
            .where((
                lick_data["event_tag"].astype(int) == SPOUT_POS_CODE
            ) & (
                lick_data["time_ms"].astype(int).isin(VALID_SPOUT_POS)
            )) \
            .ffill()    

        lick_data = lick_data.apply(pd.to_numeric)
        lick_data["mouse_id"] = os.path.basename(file_path).split("_")[3].split(".")[0]
        lick_data["cohort"] = int(file_path.split("/")[3].split()[1])
        lick_data["day"] = int(file_path.split("/")[2].split()[1])
        lick_data = lick_data.loc[lick_data["time_ms"] > 0]
        lick_data = lick_data[lick_data["event_tag"].isin(LICK_CODES + [SPOUT_EXT_CODE])]

        # Forward fill the 'start_times' for events where 'event_tag' equals the spout extension code
        start_times = lick_data["time_ms"].where(lick_data["event_tag"] == SPOUT_EXT_CODE).ffill()

        # Adjust 'time_ms' by subtracting the 'start_times' from it
        lick_data["time_ms"] = lick_data["time_ms"] - start_times

        # Infer the trial number from data ordering
        lick_data['trial_num'] = (lick_data["event_tag"] == SPOUT_EXT_CODE).astype(int)
        lick_data['trial_num'] = lick_data.groupby(['mouse_id'])['trial_num'].cumsum()

        lick_data = lick_data.loc[lick_data["time_ms"] < 5050]
        lick_data_list.append(lick_data[spout_mouse.config.LICK_DATA_COLS  + ["event_tag"]])


    lick_data_all = pd.concat(lick_data_list, ignore_index=True)
    return lick_data_all



def compute_lick_rate(lick_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lick rate and binning.

    Parameters:
        lick_data (pd.DataFrame): DataFrame containing processed lick data.

    Returns:
        pd.DataFrame: DataFrame with computed lick rates.
    """
    lick_data = lick_data.copy()
    lick_data['time_ms_binned'] = (np.ceil(lick_data['time_ms'] / BIN_SIZE_MS) * BIN_SIZE_MS).astype(int)
    lick_rate = (
        lick_data.groupby(['mouse_id', 'cohort', 'day', 'spout_id', 'trial_num', 'time_ms_binned'])
        .size()
        .reset_index(name='lick_count')
    )
    lick_rate['lick_count_hz'] = lick_rate['lick_count'] * (1000 / BIN_SIZE_MS)
    return lick_rate


def merge_spout_info(lick_rate: pd.DataFrame, spout_names: pd.DataFrame) -> pd.DataFrame:
    """
    Merge spout information into lick rate data.

    Parameters:
        lick_rate (pd.DataFrame): DataFrame containing lick rate data.
        spout_names (pd.DataFrame): DataFrame containing spout metadata.

    Returns:
        pd.DataFrame: Merged DataFrame with spout and group information.
    """
    merged_data = lick_rate.merge(spout_names, on=['cohort', 'day', 'spout_id'], how='left')
    merged_data['group'] = merged_data['mouse_id'].map(MOUSE_GROUPS)
    return merged_data
