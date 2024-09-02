# Spout Mouse

A Python package for processing and analyzing fiber photometry and lick/spout data from mouse trials.

## Features

- **Data Extraction**: Extract and organize experimental data efficiently.
- **Data Processing**: Clean and process lick and fiber photometry data robustly.
- **Visualization**: Generate insightful and publication-quality plots.
- **CLI Support**: Easily interact with the package through a command-line interface.
- **Extensible**: Modular design for easy extension and customization.

## Installation

```bash
git clone https://github.com/danielmcauley/spout_mouse.git
cd spout_mouse
pip install .
```

## Usage

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

## Contact

Daniel McAuley

Email: dmcauley4@gmail.com
GitHub: danielmcauley
