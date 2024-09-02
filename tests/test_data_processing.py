# tests/test_data_processing.py

import pytest
import pandas as pd
import os
from spout_mouse import data_processing
from unittest import mock
import tempfile
import shutil


def test_extract_zip_files_success():
    # Create temporary directory and zip file
    temp_dir = tempfile.mkdtemp()
    temp_zip_dir = tempfile.mkdtemp()
    zip_file_path = os.path.join(temp_zip_dir, 'test.zip')
    test_file_path = os.path.join(temp_zip_dir, 'test.txt')
    
    with open(test_file_path, 'w') as f:
        f.write('This is a test file.')
    
    shutil.make_archive(base_name=zip_file_path.replace('.zip', ''), format='zip', root_dir=temp_zip_dir, base_dir='.')
    
    # Test extraction
    data_processing.extract_zip_files([zip_file_path], temp_dir)
    extracted_file_path = os.path.join(temp_dir, 'test.txt')
    assert os.path.isfile(extracted_file_path)
    
    # Clean up
    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_zip_dir)


def test_extract_zip_files_file_not_found():
    with pytest.raises(FileNotFoundError):
        data_processing.extract_zip_files(['non_existent.zip'], '/tmp')


def test_load_experiment_metadata_success():
    sample_data = pd.DataFrame({
        'experiment': ['exp1', 'exp1', 'exp2'],
        'cohort': [1, 2, 1],
        'day': [1, 2, 1],
        'spout_id': [1, 2, 3],
        'spout_name': ['water', 'sucrose', 'water']
    })
    result = data_processing.load_experiment_metadata(sample_data, 'exp1', [1], [1])
    expected = pd.DataFrame({
        'cohort': [1],
        'day': [1],
        'spout_id': [1],
        'spout_name': ['water']
    })
    pd.testing.assert_frame_equal(result, expected)


def test_load_experiment_metadata_missing_columns():
    sample_data = pd.DataFrame({
        'experiment': ['exp1'],
        'cohort': [1],
        # Missing 'day', 'spout_id', 'spout_name'
    })
    with pytest.raises(ValueError):
        data_processing.load_experiment_metadata(sample_data, 'exp1', [1], [1])


def test_process_lick_data_no_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            data_processing.process_lick_data(temp_dir)


def test_compute_lick_rate():
    sample_data = pd.DataFrame({
        'mouse_id': ['mouse1'] * 3,
        'cohort': [1] * 3,
        'day': [1] * 3,
        'spout_id': [1] * 3,
        'trial_num': [1] * 3,
        'time_ms': [100, 250, 400]
    })
    result = data_processing.compute_lick_rate(sample_data)
    expected = pd.DataFrame({
        'mouse_id': ['mouse1'] * 2,
        'cohort': [1] * 2,
        'day': [1] * 2,
        'spout_id': [1] * 2,
        'trial_num': [1] * 2,
        'time_ms_binned': [200, 400],
        'lick_count': [1, 2],
        'lick_count_hz': [5.0, 10.0]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_merge_spout_info():
    lick_rate = pd.DataFrame({
        'mouse_id': ['mouse1'],
        'cohort': [1],
        'day': [1],
        'spout_id': [1],
        'trial_num': [1],
        'time_ms_binned': [200],
        'lick_count': [5],
        'lick_count_hz': [25.0]
    })
    spout_names = pd.DataFrame({
        'cohort': [1],
        'day': [1],
        'spout_id': [1],
        'spout_name': ['water']
    })
    result = data_processing.merge_spout_info(lick_rate, spout_names)
    expected = pd.DataFrame({
        'mouse_id': ['mouse1'],
        'cohort': [1],
        'day': [1],
        'spout_id': [1],
        'trial_num': [1],
        'time_ms_binned': [200],
        'lick_count': [5],
        'lick_count_hz': [25.0],
        'spout_name': ['water'],
        'group': [None]  # Assuming 'mouse1' not in MOUSE_GROUPS
    })
    pd.testing.assert_frame_equal(result, expected)
