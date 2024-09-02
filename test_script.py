import pandas as pd
from spout_mouse import (
    data_processing
)

sample_data = pd.DataFrame({
    'mouse_id': ['mouse1'] * 3,
    'cohort': [1] * 3,
    'day': [1] * 3,
    'spout_id': [1] * 3,
    'trial_num': [1] * 3,
    'time_ms': [100, 250, 400]
})

result = data_processing.compute_lick_rate(sample_data)

print(result)
