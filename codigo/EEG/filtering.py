import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import reception

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_notch_filter(data, notch_freq, fs, quality_factor=30):
    nyquist = 0.5 * fs
    notch = notch_freq / nyquist
    b, a = butter(2, [notch - notch / quality_factor, notch + notch / quality_factor], btype='bandstop')
    y = filtfilt(b, a, data)
    return y

def apply_car(data):
    mean_signal = np.mean(data, axis=0)
    return data - mean_signal

def apply_ica(data, n_components=None):
    ica = FastICA(n_components=n_components)
    components = ica.fit_transform(data.T).T
    return components, ica.mixing_

def digital_filtering(eeg_data, fs):
    try:
        # Remove power line noise
        eeg_data = butter_notch_filter(eeg_data, notch_freq=50, fs=fs)
        
        # Apply high-pass and low-pass filters (bandpass)
        eeg_data = butter_bandpass_filter(eeg_data, lowcut=0.5, highcut=50, fs=fs)
        
        # Apply Common Average Reference (CAR)
        eeg_data_car = apply_car(eeg_data)
        
        # Apply ICA to remove artifacts
        eeg_data_ica, mixing_matrix = apply_ica(eeg_data_car)
        
        return eeg_data_ica

    except Exception as e:
        print(f"An error occurred during filtering: {e}")
        return None

if __name__ == "__main__":
    eeg_data = reception.get_real_data()
    digital_filtering(eeg_data)