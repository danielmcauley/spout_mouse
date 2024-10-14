import unittest
import pandas as pd
import numpy as np
from spout_mouse import analysis
from spout_mouse.config import DOWNSAMPLE_RATE
from unittest.mock import patch


class TestAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        # Create sample data for tests
        self.lick_data_complete = pd.DataFrame({
            'mouse_id': ['1228', '1228', '1274', '1274'],  # Use string mouse IDs
            'day': [1, 1, 1, 2],
            'trial_num': [1, 2, 1, 1],
            'lick_count': [10, 15, 8, 12],
            'spout_name': ['A', 'A', 'B', 'B'],
            'group': ['sgRosa26', 'sgRosa26', 'sgRosa26', 'sgRosa26'],
            'time_ms_binned': [200, 200, 200, 200],
            'lick_count_hz': [5, 7, 4, 6]
        })

    def test_calculate_total_licks_per_trial(self):
        expected_output = pd.DataFrame({
            'mouse_id': ['1228', '1228', '1274', '1274'],
            'day': [1, 1, 1, 2],
            'trial_num': [1, 2, 1, 1],
            'lick_count_total': [10, 15, 8, 12],
            'spout_name': ['A', 'A', 'B', 'B'],
            'group': ['sgRosa26', 'sgRosa26', 'sgRosa26', 'sgRosa26']
        }).reset_index(drop=True)
        output = analysis.calculate_total_licks_per_trial(self.lick_data_complete)
        pd.testing.assert_frame_equal(output, expected_output)

    @patch('spout_mouse.config.MOUSE_GROUPS', {
        '1228': 'sgRosa26',
        '1274': 'sgRosa26'
    })
    def test_calculate_average_licks_per_spout(self):
        input_data = analysis.calculate_total_licks_per_trial(self.lick_data_complete)

        # Ensure mouse_id is of type string
        input_data['mouse_id'] = input_data['mouse_id'].astype(str)

        expected_output = pd.DataFrame({
            'mouse_id': ['1228', '1274', '1274'],
            'spout_name': ['A', 'B', 'B'],
            'day': [1, 1, 2],
            'lick_count_total': [12.5, 8.0, 12.0],
            'group': ['sgRosa26', 'sgRosa26', 'sgRosa26']
        }).reset_index(drop=True)

        output = analysis.calculate_average_licks_per_spout(input_data, combine_days=False)
        pd.testing.assert_frame_equal(output, expected_output)

    def test_organize_lick_data_by_spout(self):
        expected_output = pd.DataFrame({
            'mouse_id': ['1228', '1274', '1274'],
            'group': ['sgRosa26', 'sgRosa26', 'sgRosa26'],
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
            'group': ['sgRosa26', 'sgRosa26'],
            'spout_name': ['A', 'B'],
            'time_ms_binned': [200, 200],
            'lick_avg_all': [6.0, 5.0],
            'sem': [np.nan,  np.std([6, 4], ddof=0)/np.sqrt(2)]
        }).sort_values(by=['group', 'spout_name', 'time_ms_binned']).reset_index(drop=True)
        output = analysis.aggregate_data_and_calculate_sem(input_data)
        pd.testing.assert_frame_equal(output, expected_output)

    def test_calculate_zscores(self):
        spout_data = pd.DataFrame({
            'spout_extension_timestamp_sec': [10, 20],
            'trial_num': [1, 2],
            'mouse_id': ['1228', '1228'],
            'cohort': [1, 1],
            'day': [1, 1]
        })
        gcamp_data = pd.DataFrame({
            'timestamp_sec': np.linspace(0, 30, 300),
            'detrended_gcamp': np.random.rand(300)
        })
        result_df = analysis.calculate_zscores(spout_data, gcamp_data)
        self.assertIn('zscore_data_array', result_df.columns)
        self.assertEqual(len(result_df), 2)
        self.assertIsInstance(result_df['zscore_data_array'].iloc[0], np.ndarray)

    def test_add_auc(self):
        data = pd.DataFrame({
            'zscore_data_array': [np.random.rand(100), np.random.rand(100)]
        })
        sample_rate = 1000.0
        result_df = analysis.add_auc(data, sample_rate, DOWNSAMPLE_RATE)
        self.assertIn('auc', result_df.columns)
        self.assertEqual(len(result_df), 2)
        self.assertIsInstance(result_df['auc'].iloc[0], float)
