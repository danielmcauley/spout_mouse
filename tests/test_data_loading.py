import pytest
import numpy as np
import pandas as pd
from spout_mouse.data_loading import build_traces_df, build_spout_df
import os


def test_build_traces_df():
    downsampled_gcamp = np.random.rand(1000)
    sample_rate = 1000.0
    downsample_rate = 100
    traces_df = build_traces_df(downsampled_gcamp, sample_rate, downsample_rate)
    expected_length = len(downsampled_gcamp) - int(SEC_TO_DROP_START * sample_rate / downsample_rate) - int(SEC_TO_DROP_END * sample_rate / downsample_rate)
    assert len(traces_df) == expected_length
    assert 'timestamp_sec' in traces_df.columns
    assert 'downsampled_gcamp' in traces_df.columns


def test_build_spout_df():
    timestamps = np.array([10.0, 20.0, 30.0])
    block_path = os.path.join('path', 'to', 'cohort 1', 'day 2', 'tanks', 'block-0000-0001')
    mouse_id = '0000'
    spout_df = build_spout_df(timestamps, block_path, mouse_id)
    assert len(spout_df) == len(timestamps)
    assert all(spout_df['mouse_id'] == mouse_id)
    assert all(spout_df['cohort'] == 1)
    assert all(spout_df['day'] == 2)
    assert all(spout_df['trial_num'] == [1, 2, 3])
