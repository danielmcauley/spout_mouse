import unittest
import pandas as pd
import numpy as np
from scipy import stats
from analysis import (
    calculate_total_licks_per_trial,
    calculate_average_licks_per_spout,
    organize_lick_data_by_spout,
    aggregate_data_and_calculate_sem,
    sem,
)


class TestAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        # Create sample data for tests
        self.lick_data_complete = pd.DataFrame({
            'mouse_id': [1, 1, 2, 2],
            'day': [1, 1, 1, 2],
            'trial_num': [1, 2, 1, 1],
            'lick_count': [10, 15, 8, 12],
            'spout_name': ['A', 'A', 'B', 'B'],
            'group': ['control', 'control', 'experimental', 'experimental'],
            'time_ms_binned': [100, 200, 100, 200],
            'lick_count_hz': [5, 7, 4, 6]
        })

    def test_calculate_total_licks_per_trial(self):
        expected_output = pd.DataFrame({
            'mouse_id': [1, 1, 2],
            'day': [1, 2, 1],
            'trial_num': [1, 1, 1],
            'lick_count_total': [10, 12, 8],
            'spout_name': ['A', 'B', 'B'],
            'group': ['control', 'experimental', 'experimental']
        })
        output = calculate_total_licks_per_trial(self.lick_data_complete)
        pd.testing.assert_frame_equal(output, expected_output)

    def test_calculate_average_licks_per_spout(self):
        input_data = calculate_total_licks_per_trial(self.lick_data_complete)
        expected_output = pd.DataFrame({
            'mouse_id': [1, 2],
            'day': [1, 1],
            'spout_name': ['A', 'B'],
            'lick_count_total': [12.5, 8],
            'group': ['control', 'experimental']
        })
        output = calculate_average_licks_per_spout(input_data)
        pd.testing.assert_frame_equal(output, expected_output)

    def test_organize_lick_data_by_spout(self):
        expected_output = pd.DataFrame({
            'mouse_id': [1, 1, 2, 2],
            'group': ['control', 'control', 'experimental', 'experimental'],
            'day': [1, 1, 1, 2],
            'spout_name': ['A', 'A', 'B', 'B'],
            'time_ms_binned': [100, 200, 100, 200],
            'lick_count_avg': [5, 7, 4, 6]
        })
        output = organize_lick_data_by_spout(self.lick_data_complete)
        pd.testing.assert_frame_equal(output, expected_output)

    def test_aggregate_data_and_calculate_sem(self):
        input_data = organize_lick_data_by_spout(self.lick_data_complete)
        expected_output = pd.DataFrame({
            'group': ['control', 'experimental', 'control', 'experimental'],
            'spout_name': ['A', 'A', 'B', 'B'],
            'time_ms_binned': [100, 100, 200, 200],
            'lick_avg_all': [6, 5, 7, 6],
            'sem': [np.sqrt(2.5), np.sqrt(1), np.sqrt(1), np.sqrt(2)]
        })
        output = aggregate_data_and_calculate_sem(input_data, sem)
        pd.testing.assert_frame_equal(output, expected_output)
