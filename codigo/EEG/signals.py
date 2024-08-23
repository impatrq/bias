import numpy as np

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
