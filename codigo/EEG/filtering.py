import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import graphing # type: ignore

def digital_filtering(eeg_data, fs, notch, bandpass, car=False, ica=False):
    try:
        # Handle Nan and infinite values
        eeg_data = preprocess_data(eeg_data)

        # Print data shape
        print(f"Original data shape: {eeg_data.shape}")

        # Check the dimentions of the eeg_data
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        if notch:
            # Remove power line noise
            eeg_data = butter_notch_filter(eeg_data, notch_freq=50, fs=fs)
            print(f"Data shape after notch filter: {eeg_data.shape}")
        
        if bandpass:
            # Apply high-pass and low-pass filters (bandpass)
            eeg_data = butter_bandpass_filter(eeg_data, lowcut=0.5, highcut=50, fs=fs)
            print(f"Data shape after bandpass filter: {eeg_data.shape}")
        
        if car:
            # Apply Common Average Reference (CAR)
            eeg_data_car = apply_car(eeg_data)
            print(f"Data shape after CAR: {eeg_data_car.shape}")

            # Ensure data is still 2D after CAR
            if eeg_data_car.ndim != 2:
                raise ValueError(f"Data is not 2D after CAR application. Data shape: {eeg_data_car.shape}")
            else:
                eeg_data = eeg_data_car
        
        
        if ica:
            # Apply ICA to remove artifacts
            eeg_data_ica, mixing_matrix = apply_ica(eeg_data_car)
            print(f"Data shape after ICA: {eeg_data_ica.shape}")
            eeg_data = eeg_data_ica
        

        if eeg_data is not None:
            # Ensure the filtered data has the same length as t
            if eeg_data.shape[0] == 1:
                eeg_data = eeg_data.flatten()

            return eeg_data

    # Handle errors in the digital filtering
    except Exception as e:
        print(f"An error occurred during filtering: {e}")
        return None

# Preprocessing function to handle inf and NaN values
def preprocess_data(data):
    # Replace inf with NaN
    data = np.where(np.isinf(data), np.nan, data)
    # Remove NaN values
    data = np.nan_to_num(data)
    return data

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Bandpass filter which allows a specific range of frequencies to pass
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Range of the bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Check the padding length
    padlen = 3 * max(len(b), len(a)) 
    if data.shape[1] <= padlen:
        raise ValueError(f"The length of the input vector must be greater than padlen, which is {padlen}. Data length is {data.shape[1]}.")
    
    # Apply the bandpass filter
    y = filtfilt(b, a, data, axis=1)
    return y

def butter_notch_filter(data, notch_freq, fs, quality_factor=30):
    # Filtere use to remove a specific frequncy
    nyquist = 0.5 * fs
    notch = notch_freq / nyquist

    # Calculate the specific small band which will be filtered
    b, a = butter(2, [notch - notch / quality_factor, notch + notch / quality_factor], btype='bandstop')

    # Calculate the padding length
    padlen = 3 * max(len(b), len(a))
    if data.shape[1] <= padlen:
        raise ValueError(f"The length of the input vector must be greater than padlen, which is {padlen}. Data length is {data.shape[1]}.")
    
    # Apply the notch filter
    y = filtfilt(b, a, data, axis=1)
    return y

def apply_car(data):
    # Print the original shape of the function
    print(f"Data shape before CAR: {data.shape}")
    # Ensure mean_signal is a 2D array
    mean_signal = np.mean(data, axis=0, keepdims=True)
    # Apply CAR filter
    car_data = data - mean_signal
    return car_data

def apply_ica(data, n_components=None):
    # Apply ICA filter
    ica = FastICA(n_components=n_components)
    components = ica.fit_transform(data.T).T
    return components, ica.mixing_

def main():
    # Set parameters of eeg_data
    duration = 2
    n = 1000

    # Generate signal with pure waves
    # eeg_data = reception.get_real_combined_data(n=n, fs=n/duration, filter=True)

    # Generate random signal
    eeg_data = preprocessing.random_signal(n)

    # Generate signal with pure waves
    # eeg_data = preprocessing.pure_signal_eeg()

    # Define the sampling frequency
    fs = 500  # Hz

    # Calculate the time vector
    t = np.linspace(0, duration, n, endpoint=False)

    graphing.graph_signal_voltage_time(t, eeg_data, title="Original Signal")

    # Apply digital filtering
    filtered_data = digital_filtering(eeg_data, fs=fs, notch=True, bandpass=True, car=False, ica=False)

    # Print shape of signals
    print(f"Time vector shape: {t.shape}")
    print(f"Filtered data shape: {filtered_data.shape}")

    # Graph filtered signal
    graphing.graph_signal_voltage_time(t, filtered_data, title="Filtered Signal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()