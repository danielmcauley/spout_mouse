"""
Configuration module for spout_mouse package.
"""

# Lick Data Parameters
LICK_CODES = [331, 332, 333, 334, 335, 31, 32, 33, 34, 35]
SPOUT_EXT_CODE = 13
SPOUT_POS_CODE = 127
VALID_SPOUT_POS = [0, 1, 2, 3, 4]
LICK_DATA_COLS = ['mouse_id', 'cohort', 'day', 'spout_id', 'trial_num', 'time_ms']
NUM_TRIALS = 60
BIN_SIZE_MS = 200  # bin size for histogram of lick rate data

# Colors for Plotting
DAY_COLORS = {
    "25": "chocolate",
    "50": "teal",
    "75": "olivedrab"
}

LIQUIDS_COLORS = {
    "water": "lightskyblue",
    "05% sucrose": "cornflowerblue",
    "10% sucrose": "dodgerblue",
    "20% sucrose": "blue",
    "30% sucrose": "navy",
}

# Fiber Photometry Parameters
DOWNSAMPLE_RATE = 100  # Average every 100 samples into 1 value
TRIAL_START = 10
TRIAL_END = 15
BASELINE_START = 6  # set graph parameters relative to spout extension
BASELINE_END = 1
SEC_TO_DROP_START = 8
SEC_TO_DROP_END = 5

# Mouse Groups Mapping
MOUSE_GROUPS = {
    '1228': 'sgRosa26', 
    '1274': 'sgRosa26',
    '9694': 'sgRosa26',
    '1397': 'sgRosa26',
    '0040': 'sgRosa26',
    '0036': 'sgRosa26',
    '0037': 'sgRosa26',
    '9695': 'sgRosa26',
    '0240': 'sgTrpc6', # check number when doing operant
    '2022': 'sgRosa26',
    '0246': 'sgRosa26',
    '0247': 'sgRosa26',
    '0507': 'sgRosa26',
    '0740': 'sgRosa26',
    '0638': 'sgRosa26',
    '0640': 'sgRosa26',
    '0644': 'sgRosa26',
    '0641': 'sgRosa26',
    '1270': 'sgTrpc6',
    '1273': 'sgTrpc6',
    '9692': 'sgTrpc6',
    '0061': 'sgTrpc6',
    '0041': 'sgTrpc6',
    '0038': 'sgTrpc6',
    '0039': 'sgTrpc6',
    '0238': 'sgRosa26', # check number when doing operant
    '0239': 'sgTrpc6',
    '0245': 'sgTrpc6',
    '0248': 'sgTrpc6',
    '0509': 'sgTrpc6',
    '0637': 'sgTrpc6',
    '0639': 'sgTrpc6',
    '0645': 'sgTrpc6',
    '0066': 'sgTrpc6',
    '0067': 'sgTrpc6'
}
