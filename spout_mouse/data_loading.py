Filename: ./data_loading.py
import numpy as np
import pandas as pd
import os
from typing import List
from tdt import read_block
from .config import (
    DOWNSAMPLE_RATE,
    SEC_TO_DROP_START,
    SEC_TO_DROP_END
)


def build_traces_df(
    downsampled_gcamp: np.ndarray,
    sample_rate: float,
    downsample_rate: int = DOWNSAMPLE_RATE,
    drop_sec_start: int = SEC_TO_DROP_START,
    drop_sec_end: int = SEC_TO_DROP_END
) -> pd.DataFrame:
    """
    Build a DataFrame with downsampled GCaMP data and timestamps.

    Parameters:
        downsampled_gcamp (np.ndarray): The downsampled GCaMP data.
        sample_rate (float): The original sample rate.
        downsample_rate (int): The rate at which data was downsampled.
        drop_sec_start (int): Seconds to drop from start.
        drop_sec_end (int): Seconds to drop from end.

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp_sec', 'downsampled_gcamp'].
    """
    total_samples = len(downsampled_gcamp)
    drop_rows_start = int(drop_sec_start * sample_rate / downsample_rate)
    drop_rows_end = int(drop_sec_end * sample_rate / downsample_rate)

    traces_df = pd.DataFrame({
        'downsampled_gcamp': downsampled_gcamp,
        'timestamp_sec': (downsample_rate / sample_rate) * (np.arange(total_samples) + 1)
    })

    return traces_df.iloc[drop_rows_start:-drop_rows_end].reset_index(drop=True)


def build_spout_df(timestamps: np.ndarray, block_path: str, mouse_id: str) -> pd.DataFrame:
    """
    Build a DataFrame containing spout extension timestamps and trial numbers.

    Parameters:
        timestamps (np.ndarray): Array of spout extension timestamps in seconds.
        block_path (str): Path to the data block.
        mouse_id (str): Mouse identifier.

    Returns:
        pd.DataFrame: DataFrame with spout extension data.
    """
    spout_ext_df = pd.DataFrame({
        'spout_extension_timestamp_sec': timestamps,
        'trial_num': np.arange(1, len(timestamps) + 1),
        'mouse_id': mouse_id
    })

    # Extract cohort and day from block_path
    parts = block_path.split(os.sep)
    cohort_part = parts[3]
    cohort = int(cohort_part.split()[1])
    day_part = parts[2]
    day = int(day_part.split()[1])

    spout_ext_df['cohort'] = cohort
    spout_ext_df['day'] = day

    return spout_ext_df
