import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import filtering

def preprocess_signal(eeg_data=None, n=1000, duration=2, fs=500):

    # Time vector
    t = np.linspace(0, duration, n, endpoint=False)
    
    if eeg_data is not None:
        # Injection of real data
        signal = eeg_data
    else:
        # Stablish which signal to inject
        # signal = square_signal(t) 
        # signal = model_signal(n)
        signal = random_signal(n)
        #signal = pure_signal_eeg(duration, fs)

    signal_filtered = filtering.digital_filtering(signal, fs, notch=True, bandpass=False, car=False, ica=False)
    graph_signal_voltage_time(t, signal, title="Filtered signal")
    graph_signal_voltage_time(t, signal, title="Input signal")
    plt.tight_layout()
    plt.show()
    # signal_filtered = signal

    # Apply Fourier Transform
    signal_fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(n, d=1/fs)
    signal_fft_magnitude = np.abs(signal_fft) / n

    # Eliminate the range of negative frequencies
    signal_fft_reduced = signal_fft[:n//2]
    frequencies_reduced = frequencies[:n//2]
    signal_fft_magnitude_reduced = signal_fft_magnitude[:n//2] 

    # Apply Fourier Transform for filtered signal
    signal_fft_filtered = np.fft.fft(signal_filtered)
    frequencies_filtered = np.fft.fftfreq(n, d=1/fs)
    signal_fft_magnitude_filtered = np.abs(signal_fft_filtered) / n

    # Eliminate the range of negative frequencies
    signal_fft_reduced_filtered = signal_fft_filtered[:n//2]
    frequencies_reduced_filtered = frequencies_filtered[:n//2]
    signal_fft_magnitude_reduced_filtered = signal_fft_magnitude_filtered[:n//2] 

    # Graph the entry signal
    graph_signal_voltage_time(t, signal, title="Input signal")
    graph_signal_voltage_frequency(frequencies_reduced, signal_fft_magnitude_reduced, title='Frequency spectrum of original signal')
    graph_signal_voltage_time(t, signal_filtered, title="Input filtered signal")
    graph_signal_voltage_frequency(frequencies_reduced_filtered, signal_fft_magnitude_reduced_filtered, title='Frequency spectrum of filtered signal')

    # Set the band of frequencies for each signal
    bands = {
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
        "delta": (0.5, 4),
        "theta": (4, 8)
    }

    filtered_signals = {}

    # Plot the filtered waves in the time domain
    for band_name, band_range in bands.items():
        filtered_signals[band_name] = filter_and_reconstruct(signal_fft, frequencies, band_range)
        graph_signal_voltage_time(t, filtered_signals[band_name].real, title=f"{band_name.capitalize()} over time")

    # New sampling rate for interpolation
    new_fs = fs * 10
    new_t = np.linspace(0, duration, int(duration * new_fs), endpoint=True)

    # Interpolate each wave
    interpolated_signals = {band_name: interpolate_signal(t, sig.real, new_t) for band_name, sig in filtered_signals.items()}

    # Plot the interpolated signals
    for band_name, sig in interpolated_signals.items():
        graph_signal_voltage_time(new_t, sig, title=f"{band_name.capitalize()} interpolated")

    print(interpolated_signals)

    plt.tight_layout()
    plt.show()

    return t, interpolated_signals['alpha'], interpolated_signals['beta'], interpolated_signals['gamma'], interpolated_signals['delta'], interpolated_signals['theta']


# Create function with random values
def random_signal(n):
    # Set fixed parameter of amplitude for random EEG (uV)
    middle_amplitude = 0 
    standard_deviation = 5
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

    eeg_signal = np.tile(basic_eeg_signal, repetitions)

    return eeg_signal


# Create an eeg signal with pure waves
def pure_signal_eeg(duration=2, fs=500, alpha_amp=1, alpha_freq=10, beta_amp=2, beta_freq=20,
                   gamma_amp=3, gamma_freq=40, delta_amp=4, delta_freq=2, theta_amp=5, theta_freq=5):

    # Time vector
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Set each wave as a sine wave
    alpha_signal = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
    beta_signal = beta_amp * np.sin(2 * np.pi * beta_freq * t)
    gamma_signal = gamma_amp * np.sin(2 * np.pi * gamma_freq * t)
    delta_signal = delta_amp * np.sin(2 * np.pi * delta_freq * t)
    theta_signal = theta_amp * np.sin(2 * np.pi * theta_freq * t)
    signal = alpha_signal + beta_signal + gamma_signal + delta_signal + theta_signal

    return signal

def filter_and_reconstruct(signal_fft, frequencies, band):
    # Filter each band with the corresponding wave
    filtered_fft = np.zeros_like(signal_fft)
    band_indices = np.where((frequencies >= band[0]) & (frequencies <= band[1]))
    filtered_fft[band_indices] = signal_fft[band_indices] 
    # Reconstruct signal in the negative side
    filtered_fft[-band_indices[0]] = signal_fft[-band_indices[0]]
    # Apply inverse Fourier transform in order to obtain a signal in the time domain
    filtered_signal = np.fft.ifft(filtered_fft)
    return filtered_signal.real

def interpolate_signal(t, signal, new_t):
    # Clip new_t to the range of t to avoid out-of-bounds values
    new_t_clipped = np.clip(new_t, t.min(), t.max())

    # Interpolate the signal to new time points
    interpolated_signal = scipy.interpolate.interp1d(t, signal, kind='cubic')(new_t_clipped)
    return interpolated_signal

def graph_signal_voltage_time(t, signal, title):
    # Plot given signal in th time domain
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.grid()

def graph_signal_voltage_frequency(frequencies, magnitudes, title):
    # Plot given signal in the frquency domain
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()


if __name__ == "__main__":
    preprocess_signal()