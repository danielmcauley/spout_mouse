import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from spout_mouse.fiber_photometry import (
    nape_cart_processing,
    is_first_mouse,
    is_second_mouse,
    process_mouse,
    process_block_path,
    process_all_blocks
)
from spout_mouse.config import DOWNSAMPLE_RATE
import os


def test_nape_cart_processing():
    block_path = 'path/to/block-0000-0001-0002'
    assert nape_cart_processing(block_path) is True
    block_path = 'path/to/block-0000-0001'
    assert nape_cart_processing(block_path) is False


def test_is_first_mouse():
    block_path = 'path/to/block-0001-0000'
    assert is_first_mouse(block_path) is True
    block_path = 'path/to/block-0000-0001'
    assert is_first_mouse(block_path) is False


def test_is_second_mouse():
    block_path = 'path/to/block-0000-0001'
    assert is_second_mouse(block_path) is True
    block_path = 'path/to/block-0001-0000'
    assert is_second_mouse(block_path) is False


@patch('spout_mouse.data_loading.build_traces_df')
@patch('spout_mouse.signal_processing.detrend_signal')
@patch('spout_mouse.data_loading.build_spout_df')
@patch('spout_mouse.analysis.calculate_zscores')
@patch('spout_mouse.analysis.add_auc')
def test_process_mouse(
    mock_add_auc,
    mock_calculate_zscores,
    mock_build_spout_df,
    mock_detrend_signal,
    mock_build_traces_df
):
    downsampled_gcamp = np.random.rand(1000)
    sample_rate = 1000.0
    timestamps = np.array([10.0, 20.0, 30.0])
    block_path = 'path/to/block'
    mouse_id = '0000'
    
    mock_build_traces_df.return_value = pd.DataFrame({
        'timestamp_sec': np.linspace(0, 10, 100),
        'downsampled_gcamp': np.random.rand(100)
    })
    mock_detrend_signal.return_value = mock_build_traces_df.return_value.copy()
    mock_build_spout_df.return_value = pd.DataFrame({
        'spout_extension_timestamp_sec': timestamps,
        'trial_num': [1, 2, 3],
        'mouse_id': mouse_id,
        'cohort': 1,
        'day': 1
    })
    mock_calculate_zscores.return_value = mock_build_spout_df.return_value.copy()
    mock_add_auc.return_value = mock_build_spout_df.return_value.copy()

    result_df = process_mouse(downsampled_gcamp, sample_rate, timestamps, block_path, mouse_id)
    assert isinstance(result_df, pd.DataFrame)
    mock_build_traces_df.assert_called_once_with(downsampled_gcamp, sample_rate)
    mock_detrend_signal.assert_called_once()
    mock_build_spout_df.assert_called_once_with(timestamps, block_path, mouse_id)
    mock_calculate_zscores.assert_called_once()
    mock_add_auc.assert_called_once()


@patch('spout_mouse.fiber_photometry.read_block')
def test_process_block_path(mock_read_block):
    mock_tdt_data = MagicMock()
    mock_tdt_data.streams._470A.fs = 1000.0
    mock_tdt_data.streams._470A.data = np.random.rand(10000)
    mock_tdt_data.epocs.PtC0.onset = np.array([10.0, 20.0, 30.0])
    mock_read_block.return_value = mock_tdt_data

    block_path = 'path/to/block-0000-0001-0002'
    with patch('spout_mouse.fiber_photometry.process_mouse') as mock_process_mouse:
        mock_process_mouse.return_value = pd.DataFrame()
        result_df = process_block_path(block_path)
        mock_process_mouse.assert_called()
        assert isinstance(result_df, pd.DataFrame)
