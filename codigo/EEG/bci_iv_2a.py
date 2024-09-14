from scipy.stats import skew, kurtosis
from scipy.signal import welch
import numpy as np
from scipy.signal import butter, filtfilt
import os
import mne
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, InputLayer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = 'BCI_IV_2a/'  # Define your dataset directory

# Define frequency bands for EEG
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Step 2
def load_gdf_data(filepath):
    # Load the dataset using MNE
    raw = mne.io.read_raw_gdf(filepath, preload=True)
    # Select motor imagery (event 769 = left hand, 770 = right hand, 771 = feet, 772 = tongue)
    events, event_ids = mne.events_from_annotations(raw)
    
    # Pick EEG channels only
    raw.pick_types(eeg=True)

    # Apply bandpass filter (0.5 Hz to 100 Hz)
    raw.filter(0.5, 100)

    # Epoch the data around motor imagery events (event_id for different tasks)
    tmin, tmax = 0, 4  # Define the time window for motor imagery (0-4s)
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, event_repeated='merge')

    # Extract EEG data and corresponding labels
    X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # Labels

    return X, y

# Create a bandpass filter for a given band
def bandpass_filter(data, band, fs):
    low, high = band
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

# Apply the bandpass filter to extract different bands
def extract_bands(eeg_data, fs):
    band_data = {}
    for band_name, band_range in BANDS.items():
        band_data[band_name] = bandpass_filter(eeg_data, band_range, fs)
    return band_data

def extract_features(band_data, fs):
    """Extract scalar features for each frequency band from the EEG signal."""
    features = []

    for band_name, signal in band_data.items():
        # Check if the signal is 1D, as required for scalar functions like skew and kurtosis
        signal = np.asarray(signal).flatten()  # Ensure signal is 1D

        # Extract statistical features (ensure scalar values)
        mean_val = np.mean(signal)  # Scalar
        var_val = np.var(signal)  # Scalar
        skew_val = np.mean(skew(signal))  # Scalar, take mean in case it's an array
        kurt_val = np.mean(kurtosis(signal))  # Scalar, take mean in case it's an array
        power_val = np.sum(signal ** 2)  # Scalar (sum of squares)

        # Frequency domain feature: entropy and power spectral density
        freqs, psd = welch(signal, fs=fs)  # PSD is an array
        entropy_val = np.sum(psd * np.log(psd + 1e-10))  # Scalar entropy value

        # Summary statistics for PSD (array)
        psd_mean = np.mean(psd)  # Scalar summary of the PSD array
        psd_variance = np.var(psd)  # Variance of the PSD array

        # Combine all features into a list of scalars
        band_features = [
            mean_val, var_val, skew_val, kurt_val, power_val,
            psd_mean, psd_variance, entropy_val
        ]

        features.extend(band_features)  # Add the scalars to the feature list

    return features  # Return consistent-length feature vector

def preprocess_data(X, fs):
    """Preprocess EEG data by extracting features for each channel and frequency band."""
    features_list = []
    
    for epoch in X:
        epoch_features = []

        for ch in range(epoch.shape[0]):  # Loop over each channel
            channel_data = epoch[ch]

            # Step 1: Extract frequency bands from the channel data
            bands = extract_bands(channel_data, fs)

            # Step 2: Extract features from the bands for this channel
            channel_features = extract_features(bands, fs)
            #print(f"Channel features: {channel_features}")
            # Add the channel's features
            epoch_features.append(channel_features)

        # Print the shape of features for each epoch to
        #print(f"Shape of epoch_features: {epoch_features}")

        # Ensure all epochs produce the same feature size
        epoch_features = np.array(epoch_features)  # This line fails if inconsistent
        features_list.append(epoch_features)  # Append the features for this epoch

   # Convert the list of features to a numpy array (epochs x channels x features)
    X_processed = np.array(features_list)

    # Normalize the features
    n_epochs, n_channels, n_features = X_processed.shape
    X_processed = X_processed.reshape(n_epochs, -1)  # Flatten for normalization
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X_processed)  # Normalize
    X_processed = X_processed.reshape(n_epochs, n_channels, n_features)  # Reshape back
    X_processed = np.vstack(features_list)  # Ensure this is done after processing all epochs
    return X_processed

def build_model(input_shape, output_dimension):
    """Build a 2D CNN model that takes (channels, features) as input."""
    model = Sequential([
        # Input layer
        InputLayer(input_shape=input_shape),  # Shape: (n_channels, n_features)

        # Convolutional layer
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        # Flatten and dense layers for classification
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_dimension, activation='softmax')  # 6 output classes (forward, backward, etc.)
    ])
    # Print the model summary
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

fs = 250

# Assuming you have the load_gdf_data function defined
X_all = []
y_all = []

# List of GDF files (adjust based on the number of files you have)
GDF_FILES = ['A01T.gdf', 'A02T.gdf', 'A03T.gdf']

# Load data from each GDF file
for file in GDF_FILES:
    filepath = os.path.join(DATA_PATH, file)
    print(f"Loading data from {filepath}")
    X, y = load_gdf_data(filepath)
    X_all.append(X)
    y_all.append(y)

# Preprocess the data
X_all_processed = preprocess_data(X_all, fs)

# Flatten y_all if it's an array of arrays
y_all_flattened = np.concatenate(y_all)

# Check the shape of y_all after flattening
print(y_all_flattened.shape)
# Filter to only include classes 7, 8, 9, and 10 (representing event codes 769, 770, 771, 772)
valid_classes = [7, 8, 9, 10]
y_all_filtered = y_all_flattened[np.isin(y_all_flattened, valid_classes)]

# Reshape y_all_flattened into 2D if necessary
y_all_shaped = y_all_flattened.reshape(-1, 1)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# One-hot encode the labels
y_all_encoded = encoder.fit_transform(y_all_shaped)

print(f"Shape of X_all_processed: {X_all_processed.shape}")
print(f"Shape of y_all_encoded: {y_all_encoded.shape}")
# Convert labels to one-hot encoding
#lb = LabelBinarizer()
#y_all = lb.fit_transform(y_all)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_all_processed, y_all_encoded, test_size=0.2, random_state=42)

# Build and train the model
input_shape = (X_train.shape[1], X_train.shape[2])  # Channels, features, and 1 for "depth"
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # Add depth channel
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

model = build_model(input_shape, output_dimension=y_train.shape[1])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")