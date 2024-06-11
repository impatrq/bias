import pandas as pd
import numpy as np
import preprocessing
import reception

def extract_data(eeg_data, n, duration, fs):
    data = []
    labels = []

    # Example movements
    movements = ['forward', 'backward', 'left', 'right', 'stop']
    for movement in movements:
        for _ in range(100):  # Assume 100 samples per movement
            # Generate EEG data
            t, alpha, beta, gamma, delta, theta = preprocessing.preprocess_signal(eeg_data, n, duration, fs) # Use real data
            features = extract_features(alpha, beta, gamma, delta, theta)
            data.append(features)
            labels.append(movement)

    # Create DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels

    return df

def extract_features(alpha, beta, gamma, delta, theta):
    features = {}
    features.update(compute_features(alpha, 'alpha'))
    features.update(compute_features(beta, 'beta'))
    features.update(compute_features(gamma, 'gamma'))
    features.update(compute_features(delta, 'delta'))
    features.update(compute_features(theta, 'theta'))
    return features

def compute_features(signal, band_name):
    return {
        f'{band_name}_mean': np.mean(signal),
        f'{band_name}_variance': np.var(signal),
        f'{band_name}_skewness': np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3),
        f'{band_name}_kurtosis': np.mean((signal - np.mean(signal))**4) / (np.var(signal)**2)
    }

def main(n=1000, duration=2, fs=500):
    real_eeg_signal = reception.get_real_data(n, fs)  # Get the real EEG signal
    df = extract_data(real_eeg_signal, n, duration, fs)
    df.to_csv('extracted_features.csv', index=False)

if __name__ == "__main__":
    main()