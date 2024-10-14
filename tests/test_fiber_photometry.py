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
import warnings
from scipy.optimize import OptimizeWarning


def test_nape_cart_processing():
    block_path = 'path/to/block-0000-0001'
    assert nape_cart_processing(block_path) is True


def test_is_first_mouse():
    block_path = 'path/to/block-0000-0001'
    assert is_first_mouse(block_path) is True


def test_is_second_mouse():
    block_path = 'path/to/block-0001-0000'
    assert is_second_mouse(block_path) is True


@patch('spout_mouse.fiber_photometry.build_traces_df')
@patch('spout_mouse.fiber_photometry.detrend_signal')
@patch('spout_mouse.fiber_photometry.build_spout_df')
@patch('spout_mouse.fiber_photometry.calculate_zscores')
@patch('spout_mouse.fiber_photometry.add_auc')
def test_process_mouse(
    mock_add_auc,
    mock_calculate_zscores,
    mock_build_spout_df,
    mock_detrend_signal,
    mock_build_traces_df
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)

        downsampled_gcamp = np.random.rand(1000)
        sample_rate = 1000.0
        timestamps = np.array([10.0, 20.0, 30.0])
        block_path = 'path/to/day 1/cohort 1/block-0000-0001'
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
        mock_detrend_signal.assert_called_once_with(mock_build_traces_df.return_value)
        mock_build_spout_df.assert_called_once_with(timestamps, block_path, mouse_id)
        mock_calculate_zscores.assert_called_once_with(
            mock_build_spout_df.return_value, mock_detrend_signal.return_value
        )
        mock_add_auc.assert_called_once_with(
            mock_calculate_zscores.return_value, sample_rate, DOWNSAMPLE_RATE
        )
