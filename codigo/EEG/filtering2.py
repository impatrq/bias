import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Function to filter the data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Preprocessing function to handle inf and NaN values
def preprocess_data(data):
    # Replace inf with NaN
    data = np.where(np.isinf(data), np.nan, data)
    # Remove NaN values
    data = np.nan_to_num(data)
    return data

# Load the data
df = pd.read_csv('data.csv')

# Assume the data is in the correct format, if not, reshape accordingly
data = df.values

# Preprocess the data
data = preprocess_data(data)

# Define filter parameters
fs = 256.0  # Sampling frequency
lowcut = 1.0  # Low cut-off frequency
highcut = 50.0  # High cut-off frequency

# Apply the filter
filtered_data = np.apply_along_axis(butter_bandpass_filter, 0, data, lowcut, highcut, fs)

# Apply ICA
ica = FastICA(n_components=5)

try:
    sources = ica.fit_transform(filtered_data)
except ValueError as e:
    print(f"An error occurred during ICA fitting: {e}")

# Plot the results
plt.figure(figsize=(12, 8))
for i in range(sources.shape[1]):
    plt.subplot(sources.shape[1], 1, i + 1)
    plt.plot(sources[:, i])
    plt.title(f'ICA Component {i+1}')
plt.tight_layout()
plt.show()
