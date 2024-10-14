import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import List, Tuple
from .config import DOWNSAMPLE_RATE


def downsample_stream(data: np.ndarray, downsample_rate: int = DOWNSAMPLE_RATE) -> np.ndarray:
    """
    Downsample data by averaging every `downsample_rate` samples.

    Parameters:
        data (np.ndarray): The data to downsample.
        downsample_rate (int): The rate at which to downsample.

    Returns:
        np.ndarray: The downsampled data.
    """
    return np.array([
        np.mean(data[i: i + downsample_rate])
        for i in range(0, len(data), downsample_rate)
    ])


def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    """
    Compute a double exponential function with constant offset.

    Parameters:
        t (np.ndarray): Time vector in seconds.
        const (float): Amplitude of the constant offset.
        amp_fast (float): Amplitude of the fast component.
        amp_slow (float): Amplitude of the slow component.
        tau_slow (float): Time constant of slow component in seconds.
        tau_multiplier (float): Time constant of fast component relative to slow.

    Returns:
        np.ndarray: Computed double exponential values.
    """
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def get_bounds(traces_df: pd.DataFrame, tau_slow_min: float = 0.0001, tau_slow_max: float = 30) -> Tuple[List[float], List[float]]:
    """
    Get bounds for curve fitting parameters.

    Parameters:
        traces_df (pd.DataFrame): DataFrame containing 'downsampled_gcamp' and 'timestamp_sec'.
        tau_slow_min (float): Minimum value for slow time constant.
        tau_slow_max (float): Maximum value for slow time constant.

    Returns:
        Tuple[List[float], List[float]]: Lower and upper bounds for parameters.
    """
    signal = traces_df['downsampled_gcamp'].values
    time = traces_df['timestamp_sec'].values

    amp_min = 0
    amp_max = 2 * np.max(signal)

    time_constant_min = tau_slow_min * (time[-1] - time[0])
    time_constant_max = tau_slow_max * (time[-1] - time[0])

    offset_min = np.min(signal) if np.min(signal) < 0 else 0
    offset_max = np.max(signal)

    tau_multiplier_min = 0.01
    tau_multiplier_max = 100

    return (
        [offset_min, amp_min, amp_min, time_constant_min, tau_multiplier_min],
        [offset_max, amp_max, amp_max, time_constant_max, tau_multiplier_max]
    )


def estimate_amplitude(signal: pd.Series) -> float:
    """
    Estimate the amplitude of a signal by finding peaks and troughs.

    Parameters:
        signal (pd.Series): The signal data.

    Returns:
        float: Estimated amplitude.
    """
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)
    if peaks.size > 0 and troughs.size > 0:
        peak_amplitude = np.mean(signal.iloc[peaks])
        trough_amplitude = np.mean(signal.iloc[troughs])
        return (peak_amplitude - trough_amplitude) / 2
    else:
        return np.max(signal) / 2


def estimate_time_constant(signal: np.ndarray, timestamps: np.ndarray) -> float:
    """
    Estimate the time constant of a signal by fitting an exponential decay.

    Parameters:
        signal (np.ndarray): The signal data.
        timestamps (np.ndarray): The timestamps corresponding to the signal data.

    Returns:
        float: Estimated time constant from the exponential decay fit.
    """
    def exponential_decay(t, A, tau, offset):
        return A * np.exp(-t / tau) + offset

    initial_guess = [np.max(signal) - np.min(signal), 1.0, np.min(signal)]

    try:
        params, _ = curve_fit(exponential_decay, timestamps, signal, p0=initial_guess)
        tau = params[1]
    except RuntimeError:
        print("Curve fitting failed; using fallback value for tau.")
        tau = 1.0

    return tau


def get_initial_params(traces_df: pd.DataFrame, bounds=None) -> List[float]:
    """
    Get initial parameters for curve fitting.

    Parameters:
        traces_df (pd.DataFrame): DataFrame containing 'downsampled_gcamp' and 'timestamp_sec'.
        bounds (Tuple[List[float], List[float]], optional): Lower and upper bounds for parameters.

    Returns:
        List[float]: Initial parameters for curve fitting.
    """
    signal = traces_df['downsampled_gcamp']
    time = traces_df['timestamp_sec']

    amp_estimate = estimate_amplitude(signal)
    time_constant_estimate = estimate_time_constant(signal.values, time.values)

    initial_params = [
        np.min(signal),         # const
        amp_estimate,           # amp_fast
        amp_estimate / 2,       # amp_slow
        time_constant_estimate, # tau_slow
        0.5                     # tau_multiplier
    ]

    if bounds:
        for index, (param, lb, ub) in enumerate(zip(initial_params, *bounds)):
            if not lb <= param <= ub:
                initial_params[index] = (lb + ub) / 2

    return initial_params


def detrend_signal(traces_df: pd.DataFrame, signal_col: str = 'downsampled_gcamp', new_signal: str = 'detrended_gcamp') -> pd.DataFrame:
    """
    Detrend the signal by fitting and subtracting a double exponential curve.

    Parameters:
        traces_df (pd.DataFrame): DataFrame containing the signal data.
        signal_col (str): Column name of the signal to detrend.
        new_signal (str): Column name for the detrended signal.

    Returns:
        pd.DataFrame: DataFrame with the detrended signal added.
    """
    signal = traces_df[signal_col]
    bounds = get_bounds(traces_df)
    initial_params = get_initial_params(traces_df, bounds)

    curve_params, _ = curve_fit(
        double_exponential,
        traces_df['timestamp_sec'],
        signal,
        p0=initial_params,
        bounds=bounds,
        maxfev=5000
    )

    double_exp_fit = double_exponential(traces_df['timestamp_sec'], *curve_params)
    traces_df[new_signal] = signal - double_exp_fit

    return traces_df
