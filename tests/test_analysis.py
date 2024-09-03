import unittest
import pandas as pd
import numpy as np
from scipy import stats
from spout_mouse import analysis
from unittest.mock import patch


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
            'time_ms_binned': [200, 200, 200, 200],
            'lick_count_hz': [5, 7, 4, 6]
        })

    def test_calculate_total_licks_per_trial(self):
        expected_output = pd.DataFrame({
            'mouse_id': [1, 1, 2, 2],
            'day': [1, 1, 1, 2],
            'trial_num': [1, 2, 1, 1],
            'lick_count_total': [10, 15, 8, 12],
            'spout_name': ['A', 'A', 'B', 'B'],
            'group': ['control', 'control', 'experimental', 'experimental']
        }).reset_index(drop=True)
        output = analysis.calculate_total_licks_per_trial(self.lick_data_complete)
        pd.testing.assert_frame_equal(output, expected_output)


    @patch('spout_mouse.config.MOUSE_GROUPS', {1: 'control', 2: 'experimental'})
    def test_calculate_average_licks_per_spout(self):
        input_data = analysis.calculate_total_licks_per_trial(self.lick_data_complete)
        expected_output = pd.DataFrame({
            'mouse_id': [1, 2, 2],
            'spout_name': ['A', 'B', 'B'],
            'day': [1, 1, 2],
            'lick_count_total': [12.5, 8.0, 12.0],
            'group': ['control', 'experimental', 'experimental']
        }).reset_index(drop=True)
        output = analysis.calculate_average_licks_per_spout(input_data, False)
        pd.testing.assert_frame_equal(output, expected_output)


    def test_organize_lick_data_by_spout(self):
        expected_output = pd.DataFrame({
            'mouse_id': [1, 2, 2],
            'group': ['control', 'experimental', 'experimental'],
            'day': [1, 1, 2],
            'spout_name': ['A', 'B', 'B'],
            'time_ms_binned': [200, 200, 200],
            'lick_count_avg': [6.0, 4.0, 6.0]
        }).reset_index(drop=True)
        output = analysis.organize_lick_data_by_spout(self.lick_data_complete)
        pd.testing.assert_frame_equal(output, expected_output)


    def test_aggregate_data_and_calculate_sem(self):
        input_data = analysis.organize_lick_data_by_spout(self.lick_data_complete)
        expected_output = pd.DataFrame({
            'group': ['control', 'experimental'],
            'spout_name': ['A', 'B'],
            'time_ms_binned': [200, 200],
            'lick_avg_all': [6.0, 5.0],
            'sem': [np.nan,  np.std([6, 4], ddof=0)/np.sqrt(2)]
        }).sort_values(by=['group', 'spout_name', 'time_ms_binned']).reset_index(drop=True)
        output = analysis.aggregate_data_and_calculate_sem(input_data)
        pd.testing.assert_frame_equal(output, expected_output)
