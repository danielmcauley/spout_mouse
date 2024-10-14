# Spout Mouse

A Python package for processing and analyzing fiber photometry and lick/spout data from mouse trials.

## Features

- **Data Extraction**: Extract and organize experimental data efficiently.
- **Data Processing**: Clean and process lick and fiber photometry data robustly.
- **Visualization**: Generate insightful and publication-quality plots.
- **CLI Support**: Easily interact with the package through a command-line interface.
- **Extensible**: Modular design for easy extension and customization.

## Modules
The `spout_mouse` package is organized into several modules for clarity and maintainability:

- `fiber_photometry`: High-level functions for processing fiber photometry data.
- `signal_processing`: Functions related to signal processing tasks like downsampling and detrending.
- `data_loading`: Functions for loading and building data structures from raw data.
- `analysis`: Functions that perform calculations like z-scores and area under the curve (AUC).
- `plotting`: Functions for generating plots.
- `utils`: Utility functions for tasks like authorizing Google Sheets access.

## Installation

```bash
git clone https://github.com/danielmcauley/spout_mouse.git
cd spout_mouse
pip install .
```

## Usage

### Processing Lick Data

```python
import json
import pandas as pd
from spout_mouse import (
    authorize_google_sheets,
    get_experiment_data,
    extract_zip_files,
    load_experiment_metadata,
    process_lick_data,
    compute_lick_rate,
    merge_spout_info,
    plot_lick_rate,
    plot_total_licks,
)

# Load Google Sheets credentials
with open('path_to_credentials.json') as f:
    credentials_json = f.read()

# Authorize Google Sheets client
client = authorize_google_sheets(credentials_json)

# Fetch experiment data
experiment_records = get_experiment_data(client)
experiment_df = pd.DataFrame(experiment_records)

# Load and process spout metadata
spout_names = load_experiment_metadata(
    experiment_df,
    experiment_name='sucrosesredo WR',
    cohorts=[1, 2, 3, 4],
    days=[3, 4, 5]
)

# Extract data files
extract_zip_files(['path_to_zip1.zip', 'path_to_zip2.zip'], 'extracted_data/')

# Process lick data
lick_data = process_lick_data('extracted_data/')
lick_rate = compute_lick_rate(lick_data)
merged_data = merge_spout_info(lick_rate, spout_names)

# Generate plots
lick_rate_plot = plot_lick_rate(merged_data)
total_licks_plot = plot_total_licks(merged_data)

# Save plots
lick_rate_plot.save('lick_rate_plot.png')
total_licks_plot.save('total_licks_plot.png')

```

### Processing Fiber Photometry Data

```python
import pandas as pd
from spout_mouse import (
    process_all_blocks,
    prepare_fp_dataframe,
    clean_fp_trials,
    truncate_zscore_arrays,
    merge_fp_with_lick_data,
    calculate_mean_zscore,
    prepare_long_format,
    calculate_mean_sem_zscores,
    plot_zscore_traces,  # Assuming this function exists for plotting
)

# Define the directory pattern for the data blocks
directory_pattern = 'path/to/data/blocks/*'

# Process all data blocks matching the pattern
processed_data = process_all_blocks(directory_pattern)

# Prepare the fiber photometry DataFrame
fp_df_prepared = prepare_fp_dataframe(processed_data)

# Clean trials to ensure consistent number per mouse/day
fp_df_clean = clean_fp_trials(fp_df_prepared, num_trials=60)

# Truncate z-score arrays to the minimum length
fp_df_truncated = truncate_zscore_arrays(fp_df_clean)

# Assume you have lick data loaded in `lick_data_complete`
# Merge fiber photometry data with lick data to add spout information
fp_df_complete = merge_fp_with_lick_data(fp_df_truncated, lick_data_complete)

# Calculate mean z-score per mouse, day, and spout
mean_zscore_df = calculate_mean_zscore(fp_df_complete, across_days=False)

# Prepare data in long format for plotting
mean_zscore_long = prepare_long_format(mean_zscore_df, across_days=False)

# Calculate mean and SEM of z-scores for plotting
mean_sem_zscores = calculate_mean_sem_zscores(mean_zscore_df, across_days=False)

# Generate plots (assuming you have a plotting function)
plot_zscore_traces(mean_sem_zscores)

# Save the processed data
mean_sem_zscores.to_csv('mean_sem_zscores.csv', index=False)
```

**Note:** Replace `'path/to/data/blocks/*'` with the actual path to your data blocks. Ensure that lick_data_complete is prepared and available when merging with fiber photometry data.

## Contact

Daniel McAuley

Email: dmcauley4@gmail.com
GitHub: danielmcauley
