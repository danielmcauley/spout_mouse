Filename: ./fiber_photometry.py
import os
import glob
import pandas as pd
from tqdm import tqdm
from tdt import read_block
from .config import (
    MOUSE_GROUPS,
    DOWNSAMPLE_RATE,
)
from .signal_processing import downsample_stream
from .data_loading import build_traces_df, build_spout_df
from .analysis import detrend_signal, calculate_zscores, add_auc


def nape_cart_processing(block_path: str) -> bool:
    """
    Determine if the block path indicates nape cart processing.

    Parameters:
        block_path (str): Path to the data block.

    Returns:
        bool: True if nape cart processing, False otherwise.
    """
    parts = os.path.basename(block_path).split("-")
    return len(parts) == 3


def is_first_mouse(block_path: str) -> bool:
    """
    Check if the block corresponds to the first mouse.

    Parameters:
        block_path (str): Path to the data block.

    Returns:
        bool: True if first mouse, False otherwise.
    """
    parts = os.path.basename(block_path).split("-")
    return parts[1] == '0000'


def is_second_mouse(block_path: str) -> bool:
    """
    Check if the block corresponds to the second mouse.

    Parameters:
        block_path (str): Path to the data block.

    Returns:
        bool: True if second mouse, False otherwise.
    """
    parts = os.path.basename(block_path).split("-")
    return parts[0] == '0000'


def process_mouse(
    downsampled_gcamp: np.ndarray,
    sample_rate: float,
    timestamps: np.ndarray,
    block_path: str,
    mouse_id: str
) -> pd.DataFrame:
    """
    Process data for a single mouse.

    Parameters:
        downsampled_gcamp (np.ndarray): Downsampled GCaMP data.
        sample_rate (float): Original sample rate.
        timestamps (np.ndarray): Spout extension timestamps.
        block_path (str): Path to the data block.
        mouse_id (str): Mouse identifier.

    Returns:
        pd.DataFrame: Processed data for the mouse.
    """
    traces_df = build_traces_df(downsampled_gcamp, sample_rate)
    gcamp_detrended_df = detrend_signal(traces_df)
    spout_ext_df = build_spout_df(timestamps, block_path, mouse_id)
    joined_df = calculate_zscores(spout_ext_df, gcamp_detrended_df)
    joined_df = add_auc(joined_df, sample_rate, DOWNSAMPLE_RATE)
    return joined_df


def process_block_path(block_path: str) -> pd.DataFrame:
    """
    Process a data block and return processed data.

    Parameters:
        block_path (str): Path to the data block.

    Returns:
        pd.DataFrame: Processed data from the block.
    """
    tdt_data = read_block(block_path)

    if nape_cart_processing(block_path):
        sample_rate = tdt_data.streams._470A.fs
        downsampled_gcamp = downsample_stream(tdt_data.streams._470A.data)
        timestamps = tdt_data.epocs.PtC0.onset
        mouse_id = str(os.path.basename(block_path).split("-")[0])
        processed_mice = process_mouse(downsampled_gcamp, sample_rate, timestamps, block_path, mouse_id)
    elif is_first_mouse(block_path):
        sample_rate = tdt_data.streams._465A.fs
        downsampled_gcamp = downsample_stream(tdt_data.streams._465A.data)
        timestamps = tdt_data.epocs.PtC0.onset
        mouse_id = str(os.path.basename(block_path).split("-")[0])
        processed_mice = process_mouse(downsampled_gcamp, sample_rate, timestamps, block_path, mouse_id)
    elif is_second_mouse(block_path):
        sample_rate = tdt_data.streams._465C.fs
        downsampled_gcamp = downsample_stream(tdt_data.streams._465C.data)
        timestamps = tdt_data.epocs.PtC2.onset
        mouse_id = str(os.path.basename(block_path).split("-")[1])
        processed_mice = process_mouse(downsampled_gcamp, sample_rate, timestamps, block_path, mouse_id)
    else:
        # Process first mouse
        sample_rate = tdt_data.streams._465A.fs
        downsampled_gcamp = downsample_stream(tdt_data.streams._465A.data)
        timestamps = tdt_data.epocs.PtC0.onset
        mouse_id = str(os.path.basename(block_path).split("-")[0])
        processed_mouse_1 = process_mouse(downsampled_gcamp, sample_rate, timestamps, block_path, mouse_id)
        # Process second mouse
        sample_rate = tdt_data.streams._465C.fs
        downsampled_gcamp = downsample_stream(tdt_data.streams._465C.data)
        timestamps = tdt_data.epocs.PtC2.onset
        mouse_id = str(os.path.basename(block_path).split("-")[1])
        processed_mouse_2 = process_mouse(downsampled_gcamp, sample_rate, timestamps, block_path, mouse_id)
        processed_mice = pd.concat([processed_mouse_1, processed_mouse_2])

    return processed_mice


def process_all_blocks(directory_pattern: str) -> pd.DataFrame:
    """
    Process all data blocks matching a directory pattern.

    Parameters:
        directory_pattern (str): Glob pattern to match block paths.

    Returns:
        pd.DataFrame: Concatenated DataFrame of processed data from all blocks.
    """
    processed_blocks = []
    block_paths = glob.glob(directory_pattern)
    for block_path in tqdm(block_paths, desc="Processing Blocks"):
        processed_blocks.append(process_block_path(block_path))
    return pd.concat(processed_blocks, ignore_index=True)
