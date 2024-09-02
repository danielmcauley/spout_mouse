# tests/test_plotting.py

import pytest
import pandas as pd
from plotnine import ggplot
from spout_mouse import plotting


def test_plot_lick_rate():
    sample_data = pd.DataFrame({
        'time_ms_binned': [200, 400, 600],
        'lick_avg_all': [5, 10, 7],
        'sem': [0.5, 0.8, 0.6],
        'spout_name': ['water', 'water', 'water'],
        'group': ['group1', 'group1', 'group1']
    })
    plot = plotting.plot_lick_rate(sample_data)
    assert isinstance(plot, ggplot)


def test_plot_total_licks():
    sample_data = pd.DataFrame({
        'spout_name': ['water', 'sucrose', 'water'],
        'lick_count_total': [50, 70, 65],
        'group': ['group1', 'group1', 'group2']
    })
    plot = plotting.plot_total_licks(sample_data)
    assert isinstance(plot, ggplot)
