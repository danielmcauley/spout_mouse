import unittest
import pandas as pd
import numpy as np
from spout_mouse import analysis
from spout_mouse.config import DOWNSAMPLE_RATE, MOUSE_GROUPS
from unittest.mock import patch


class TestAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        # Create sample data for tests
        self.lick_data_complete = pd.DataFrame({
            'mouse_id': ['1228', '1228', '1274', '1274'],
            'day': [1, 1, 1, 2],
            'trial_num': [1, 2, 1, 1],
            'lick_count': [10, 15, 8, 12],
            'spout_name': ['A', 'A', 'B', 'B'],
            'group': ['sgRosa26', 'sgRosa26', 'sgRosa26', 'sgRosa26'],
            'time_ms_binned': [200, 200, 200, 200],
            'lick_count_hz': [5, 7, 4, 6]
        })

        # Sample data for fp_df
        self.fp_df = pd.DataFrame({
            'mouse_id': ['1228', '1228', '1274', '1274', '0037', '0039'],
            'cohort': [1, 1, 1, 1, 2, 2],
            'day': [1, 1, 1, 1, 1, 1],
            'trial_num': [1, 2, 1, 2, 1, 2],
            'zscore_data_array': [np.random.rand(100) for _ in range(6)]
        })
        # Assign 'group' column arbitrarily
        self.fp_df['group'] = ['sgRosa26', 'sgRosa26', 'sgRosa26', 'sgRosa26', 'control', 'control']

        # Sample data for lick_data
        self.lick_data = pd.DataFrame({
            'mouse_id': ['1228', '1228', '1274', '1274', '0037', '0039'],
            'cohort': [1, 1, 1, 1, 2, 2],
            'day': [1, 1, 1, 1, 1, 1],
            'trial_num': [1, 2, 1, 2, 1, 2],
            'spout_name': ['water', 'sucrose', 'water', 'sucrose', 'water', 'sucrose']
        })

        # Sample data for calculate_auc_by_mouse_spout
        self.fp_auc_data = pd.DataFrame({
            'group': ['sgRosa26', 'sgRosa26', 'sgRosa26', 'sgRosa26', 'control', 'control']
            'day': [1, 1, 2, 1, 1, 2],
            'mouse_id': ['1228', '1228', '1274', '1274', '0037', '0039'],
            'spout_name': ['water', 'water', 'sucrose', 'water', 'water', 'sucrose'],
            'auc': [0.5, 0.7, 0.8, 0.6, 0.7, 0.9]
        })

        # Sample data for calculate_auc_by_mouse_spout
        self.fp_auc_data = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'day': [1, 1, 2, 1, 1, 2],
            'mouse_id': [1, 1, 1, 2, 2, 2],
            'spout_name': ['left', 'left', 'left', 'right', 'right', 'right'],
            'auc': [0.5, 0.7, 0.8, 0.6, 0.7, 0.9]
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

    @patch('spout_mouse.config.MOUSE_GROUPS', {'1274': 'sgRosa26'})
    def test_prepare_fp_dataframe(self):
        excluded_mice = ["0037", "9694", "1228", "0036", "0039", "9692", "0061"]
        prepared_df = analysis.prepare_fp_dataframe(self.fp_df, excluded_mice)
        # Check that mouse_id is string
        self.assertTrue(prepared_df['mouse_id'].dtype == object)
        # Check that 'group' is mapped correctly
        expected_groups = prepared_df['mouse_id'].map(MOUSE_GROUPS)
        pd.testing.assert_series_equal(prepared_df['group'], expected_groups, check_names=False)
        # Check that excluded mice are removed
        self.assertFalse(prepared_df['mouse_id'].isin(excluded_mice).any())

    def test_clean_fp_trials(self):
        # Assume we want to keep only 1 trial per mouse for simplicity
        cleaned_df = analysis.clean_fp_trials(self.fp_df, num_trials=1)
        # Check that each mouse has only 1 trial
        counts = cleaned_df.groupby(['cohort', 'day', 'mouse_id']).size()
        self.assertTrue((counts == 1).all())
        # Check that trial numbers are reset starting from 1
        expected_trial_nums = [1] * len(cleaned_df)
        self.assertListEqual(cleaned_df['trial_num'].tolist(), expected_trial_nums)

    def test_truncate_zscore_arrays(self):
        # Ensure 'zscore_data_array' column is of type 'object'
        self.fp_df['zscore_data_array'] = self.fp_df['zscore_data_array'].astype(object)
        
        # Assign arrays of different lengths
        self.fp_df.at[0, 'zscore_data_array'] = np.random.rand(90)
        self.fp_df.at[1, 'zscore_data_array'] = np.random.rand(80)
        
        truncated_df = analysis.truncate_zscore_arrays(self.fp_df)
        # Check that all arrays have the same length
        lengths = truncated_df['zscore_data_array'].apply(len)
        self.assertTrue((lengths == lengths.iloc[0]).all())

    def test_merge_fp_with_lick_data(self):
        merged_df = analysis.merge_fp_with_lick_data(self.fp_df, self.lick_data)
        # Check that 'spout_name' is added
        self.assertIn('spout_name', merged_df.columns)
        # Check that the number of rows matches
        self.assertEqual(len(merged_df), len(self.fp_df))
        # Check for NaN values in 'spout_name' (should be none if data aligns)
        self.assertFalse(merged_df['spout_name'].isna().any())

    def test_calculate_mean_zscore(self):
        # Assume 'fp_df_complete' is the merged DataFrame
        fp_df_complete = analysis.merge_fp_with_lick_data(self.fp_df, self.lick_data)
        mean_zscore_df = analysis.calculate_mean_zscore(fp_df_complete, across_days=False)
        # Check that the mean zscore_data_array is calculated
        self.assertIn('zscore_data_array', mean_zscore_df.columns)
        # Check that group mapping is correct
        self.assertIn('group', mean_zscore_df.columns)
        # Check that the number of unique combinations matches
        expected_groups = fp_df_complete.groupby(['mouse_id', 'day', 'spout_name']).size().reset_index()
        self.assertEqual(len(mean_zscore_df), len(expected_groups))

    def test_prepare_long_format(self):
        fp_df_complete = analysis.merge_fp_with_lick_data(self.fp_df, self.lick_data)
        mean_zscore_df = analysis.calculate_mean_zscore(fp_df_complete, across_days=False)
        mean_zscore_long = analysis.prepare_long_format(mean_zscore_df, across_days=False)
        # Check that DataFrame is in long format
        self.assertIn('time', mean_zscore_long.columns)
        # Check that 'zscore_data_array' is a float
        self.assertTrue(mean_zscore_long['zscore_data_array'].dtype == float)
        # Check that 'group_name' column exists
        self.assertIn('group_name', mean_zscore_long.columns)

    def test_calculate_mean_sem_zscores(self):
        fp_df_complete = analysis.merge_fp_with_lick_data(self.fp_df, self.lick_data)
        mean_zscore_df = analysis.calculate_mean_zscore(fp_df_complete, across_days=False)
        mean_sem_zscores = analysis.calculate_mean_sem_zscores(mean_zscore_df, across_days=False)
        # Check that 'mean_zscore' and 'sem_zscore' columns exist
        self.assertIn('mean_zscore', mean_sem_zscores.columns)
        self.assertIn('sem_zscore', mean_sem_zscores.columns)
        # Check that 'time' column exists
        self.assertIn('time', mean_sem_zscores.columns)
        # Check that the DataFrame is not empty
        self.assertFalse(mean_sem_zscores.empty)

    def test_calculate_auc_by_mouse_spout(self):
        # Test case when across_days=False
        result = analysis.calculate_auc_by_mouse_spout(self.fp_auc_data, across_days=False)
        # Expected output for across_days=False
        expected_output = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'day': [1, 2, 1, 2],
            'mouse_id': [1, 1, 2, 2],
            'spout_name': ['left', 'left', 'right', 'right'],
            'mean': [0.6, 0.8, 0.65, 0.9],
            'sem': [0.1, np.nan, 0.05, np.nan]
        })
        pd.testing.assert_frame_equal(result, expected_output, check_exact=False, check_less_precise=True)
        # Test case when across_days=True
        result_across_days = analysis.calculate_auc_by_mouse_spout(self.fp_auc_data, across_days=True)
        # Expected output for across_days=True
        expected_output_across_days = pd.DataFrame({
            'group': ['A', 'B'],
            'mouse_id': [1, 2],
            'spout_name': ['left', 'right'],
            'mean': [0.6666666667, 0.7333333333],
            'sem': [0.0881928269, 0.0881928269]
        })
        pd.testing.assert_frame_equal(result_across_days, expected_output_across_days, check_exact=False, check_less_precise=True)
        
