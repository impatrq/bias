import pandas as pd
import numpy as np
import preprocessing
import reception
from mne import io, Epochs, pick_types
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

def load_online_data(subject=1):
    # Load data from an online dataset (example using EEG Motor Movement/Imagery Dataset)
    raw_files = eegbci.load_data(subject=subject, runs=[3, 7, 11])  # Load sample subject data
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_files])
    return raw

def load_offline_data(eeg_data, n, duration, fs):
    # Load offline data, typically from a local EEG recording device
    t, alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(eeg_data, n, duration, fs)
    return alpha, beta, gamma, delta, theta

def extract_data(eeg_data, labels, n, duration, fs, online=True):
    data = []

    if online:
        # Process and extract data from the raw signal for online data
        events = mne.find_events(eeg_data, shortest_event=0, stim_channel='STI 014')
        event_id = dict(hands=2, feet=3)
        picks = pick_types(eeg_data.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        
        # Create epochs around the events
        epochs = Epochs(eeg_data, events, event_id, tmin=-0.2, tmax=2, proj=True, picks=picks, baseline=None, preload=True)

        for event in events:
            if event[2] in event_id.values():
                alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(epochs, n, duration, fs)
                features = extract_features(alpha, beta, gamma, delta, theta)
                features['label'] = labels[event[2] - 2]  # Adjust label index to match your labels
                data.append(features)
    else:
        for label in labels:
            for _ in range(100):  # Assume 100 samples per movement
                # Generate EEG data
                alpha, beta, gamma, delta, theta = load_offline_data(eeg_data, n, duration, fs) # Use real data
                features = extract_features(alpha, beta, gamma, delta, theta)
                features['label'] = label
                data.append(features)

    # Create DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels

    return df

def extract_features(alpha, beta, gamma, delta, theta):
    features = {}
    for band_name, signal in zip(['alpha', 'beta', 'gamma', 'delta', 'theta'], [alpha, beta, gamma, delta, theta]):
            features.update(compute_features(signal, band_name))
    return features

def compute_features(signal, band_name):
    return {
        f'{band_name}_mean': np.mean(signal),
        f'{band_name}_variance': np.var(signal),
        f'{band_name}_skewness': np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3),
        f'{band_name}_kurtosis': np.mean((signal - np.mean(signal))**4) / (np.var(signal)**2),
        f'{band_name}_entropy': spectral_entropy(signal),
        f'{band_name}_bandpower_ratio': bandpower_ratio(signal, band_name)
    }

def spectral_entropy(signal):
    # Compute spectral entropy
    psd = np.abs(np.fft.fft(signal))**2
    psd_norm = psd / psd.sum()
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return entropy

def bandpower_ratio(signal, band_name):
    # Compute bandpower ratio (example with total power)
    total_power = np.sum(np.abs(signal)**2)
    band_power = np.sum(np.abs(signal)**2)
    return band_power / total_power

def main(n=1000, duration=2, fs=500, online=True):
    labels = ['forward', 'backward', 'left', 'right', 'stop', 'rest']
    if online:
        raw = load_online_data()

    else:
        real_eeg_signal = reception.get_real_data(n, fs)  # Get the real EEG signal for offline data
        raw = real_eeg_signal
    
    df = extract_data(raw, labels, n, duration, fs, online=online)
    df.to_csv('extracted_features.csv', index=False)

if __name__ == "__main__":
    main()