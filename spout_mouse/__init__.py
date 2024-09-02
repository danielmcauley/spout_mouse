from .config import *
from .utils import authorize_google_sheets, get_experiment_data
from .data_processing import (
    extract_zip_files,
    load_experiment_metadata,
    process_lick_data,
    compute_spout_order,
    compute_lick_rate,
    create_and_merge_spine,
    fill_missing_spout_ids,
    merge_spout_info,
)
from .analysis import (
    calculate_total_licks_per_trial,
    calculate_average_licks_per_spout,
    organize_lick_data_by_spout,
    aggregate_data_and_calculate_sem,
)
from .plotting import plot_lick_rate, plot_total_licks

__all__ = [
    'authorize_google_sheets',
    'get_experiment_data',
    'extract_zip_files',
    'load_experiment_metadata',
    'process_lick_data',
    'compute_lick_rate',
    'merge_spout_info',
    'plot_lick_rate',
    'plot_total_licks',
    'LICK_CODES',
    'SPOUT_EXT_CODE',
    'SPOUT_POS_CODE',
    'VALID_SPOUT_POS',
    'LICK_DATA_COLS',
    'NUM_TRIALS',
    'BIN_SIZE_MS',
    'DAY_COLORS',
    'LIQUIDS_COLORS',
    'DOWNSAMPLE_RATE',
    'TRIAL_START',
    'TRIAL_END',
    'BASELINE_START',
    'BASELINE_END',
    'SEC_TO_DROP_START',
    'SEC_TO_DROP_END',
    'GOOGLE_SHEET_URL',
    'MOUSE_GROUPS',
]
