import pytest
import numpy as np
import pandas as pd
from spout_mouse.signal_processing import (
    downsample_stream,
    double_exponential,
    get_bounds,
    estimate_amplitude,
    estimate_time_constant,
    get_initial_params,
    detrend_signal
)
from scipy.optimize import curve_fit


def test_downsample_stream():
    data = np.arange(1000)
    downsample_rate = 100
    downsampled = downsample_stream(data, downsample_rate=downsample_rate)
    expected = np.arange(49.5, 1000, downsample_rate)
    assert len(downsampled) == 10
    np.testing.assert_allclose(downsampled, expected)


def test_double_exponential():
    t = np.linspace(0, 10, 100)
    params = [1.0, 2.0, 3.0, 0.5, 2.0]
    result = double_exponential(t, *params)
    assert result.shape == t.shape
    assert not np.isnan(result).any()


def test_get_bounds():
    traces_df = pd.DataFrame({
        'downsampled_gcamp': np.random.rand(100),
        'timestamp_sec': np.linspace(0, 10, 100)
    })
    lower_bounds, upper_bounds = get_bounds(traces_df)
    assert len(lower_bounds) == 5
    assert len(upper_bounds) == 5
    assert all(l <= u for l, u in zip(lower_bounds, upper_bounds))


def test_estimate_amplitude():
    signal = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 100)))
    amplitude = estimate_amplitude(signal)
    assert amplitude > 0
    assert amplitude <= 1


def test_estimate_time_constant():
    t = np.linspace(0, 10, 100)
    signal = np.exp(-t / 2.0)
    tau = estimate_time_constant(signal, t)
    assert np.isclose(tau, 2.0, atol=0.5)


def test_get_initial_params():
    traces_df = pd.DataFrame({
        'downsampled_gcamp': np.exp(-np.linspace(0, 5, 100)),
        'timestamp_sec': np.linspace(0, 5, 100)
    })
    initial_params = get_initial_params(traces_df)
    assert len(initial_params) == 5
    assert all(isinstance(param, float) for param in initial_params)


def test_detrend_signal():
    traces_df = pd.DataFrame({
        'downsampled_gcamp': np.exp(-np.linspace(0, 10, 100)) + np.random.normal(0, 0.01, 100),
        'timestamp_sec': np.linspace(0, 10, 100)
    })
    detrended_df = detrend_signal(traces_df)
    assert 'detrended_gcamp' in detrended_df.columns
    assert len(detrended_df['detrended_gcamp']) == 100
    # Check that the mean of detrended signal is close to zero
    assert np.isclose(detrended_df['detrended_gcamp'].mean(), 0, atol=0.1)
