import numpy as np
import random

# Create function with random values
def random_signal(n):
    # Set fixed parameter of amplitude for random EEG (uV)
    middle_amplitude = 0 
    standard_deviation = 5 # uV
    return np.random.normal(middle_amplitude, standard_deviation, n)

def square_signal(t):
    # Set parameters of square wave
    amplitude = 3
    frequency = 10

    # Save and return square wave
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

def model_signal(n):
    # Create a model eeg signal
    basic_eeg_signal = np.array([
        0.5, 0.4, 0.3, 0.2, 0.1,
        -0.1, -0.2, -0.3, -0.4, -0.5,
        0.8, 0.7, 0.6, 0.5, 0.4,
        -0.4, -0.5, -0.6, -0.7, -0.8,
        0.3, 0.2, 0.1, 0, -0.1,
        -0.2, -0.3, -0.4, -0.5, -0.6,
        0.7, 0.6, 0.5, 0.4, 0.3,
        -0.3, -0.4, -0.5, -0.6, -0.7,
        0.2, 0.1, 0, -0.1, -0.2,
        -0.3, -0.4, -0.5, -0.6, -0.7,
    ])


    eeg_signal = []

    # Repeat it until you have reached the number of samples
    repetitions = n // len(basic_eeg_signal)

    # Add repetitions to the signal
    eeg_signal = np.tile(basic_eeg_signal, repetitions)

    return eeg_signal

# Create an eeg signal with pure waves
def pure_signal_eeg(n, fs, alpha_amp=1, alpha_freq=10, beta_amp=2, beta_freq=20,
                    gamma_amp=3, gamma_freq=40, delta_amp=4, delta_freq=2, theta_amp=5, theta_freq=5):
    duration = n / fs
    # Time vector
    t = np.linspace(0, duration, n, endpoint=False)

    # Set each wave as a sine wave
    alpha_signal = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
    beta_signal = beta_amp * np.sin(2 * np.pi * beta_freq * t)
    gamma_signal = gamma_amp * np.sin(2 * np.pi * gamma_freq * t)
    delta_signal = delta_amp * np.sin(2 * np.pi * delta_freq * t)
    theta_signal = theta_amp * np.sin(2 * np.pi * theta_freq * t)
    signal = alpha_signal + beta_signal + gamma_signal + delta_signal + theta_signal

    return signal


def generate_synthetic_eeg_bandpower(n_samples, n_channels, fs, command=None):
    """
    Generate synthetic raw EEG data for multiple channels.
    The output is a dictionary where each channel has 1000 raw samples.
    Simulate different tasks by altering the signal patterns.
    """
    t = np.linspace(0, n_samples/fs, n_samples, endpoint=False)
    data = {}

    for ch in range(n_channels):
        # Simulate different frequency bands with some basic correlations
        base_alpha = np.sin(2 * np.pi * 10 * t)  # Alpha signal_wave (10 Hz)
        base_beta = np.sin(2 * np.pi * 20 * t)   # Beta signal_wave (20 Hz)
        base_theta = np.sin(2 * np.pi * 6 * t)   # Theta signal_wave (6 Hz)
        base_delta = np.sin(2 * np.pi * 2 * t)   # Delta signal_wave (2 Hz)
        base_gamma = np.sin(2 * np.pi * 40 * t)  # Gamma signal_wave (40 Hz)

        alpha_power = 1.0
        beta_power = 1.0
        theta_power = 1.0
        delta_power = 1.0
        gamma_power = 1.0 # Adjust signal based on the command

        if command == "forward":
            alpha_power = 1.5
            beta_power = 0.5
        elif command == "backward":
            alpha_power = 0.5
            beta_power = 1.5
        elif command == "left":
            theta_power = 1.5
            delta_power = 0.5
        elif command == "right":
            theta_power = 0.5
            delta_power = 1.5
        elif command == "stop":
            alpha_power = 0.2
            beta_power = 0.2
            gamma_power = 0.2
        else:  # rest
            alpha_power = 1.0
            beta_power = 1.0
            theta_power = 1.0
            delta_power = 1.0
            gamma_power = 1.0        

        # Generate signal with some added randomness and correlations
        signal = (
            alpha_power * base_alpha +
            beta_power * base_beta +
            theta_power * base_theta +
            delta_power * base_delta +
            gamma_power * base_gamma
        )

        # Add channel correlation (e.g., 10% of the previous channel’s signal)
        if ch > 0:
            signal += 0.1 * data[ch-1]

        # Add random noise to simulate realistic EEG signals
        noise = np.random.normal(0, 0.1, size=t.shape)
        signal += noise

        # Store the raw signal in the dictionary
        data[ch] = signal

    return data

def generate_synthetic_eeg(n_samples, n_channels, fs):
    """
    Generate synthetic raw EEG data for multiple channels. 
    The output is a dictionary where each channel has 1000 raw samples.
    """
    t = np.linspace(0, n_samples/fs, n_samples, endpoint=False)
    data = {}

    for ch in range(n_channels):
        # Create a raw EEG signal by summing several sine waves to simulate brain activity
        signal = (
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(8, 13) * t) +  # Simulate alpha signal_wave (8-13 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(13, 30) * t) +  # Simulate beta signal_wave (13-30 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(4, 8) * t) +   # Simulate theta signal_wave (4-8 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(1, 4) * t) +   # Simulate delta signal_wave (0.5-4 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(0, 50) * t)    # Simulate gamma signal_wave (30-100 Hz)
        )

        # Add random noise to simulate realistic EEG signals
        noise = np.random.normal(0, 0.5, size=t.shape)
        signal += noise

        # Store the raw signal in the dictionary
        data[ch] = signal

    return data
