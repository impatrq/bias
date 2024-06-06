import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

def main():
    # Set sampling frequency
    fs = 500
    # Number of samples
    N = 1000

    # Set duration of signal
    duration = N / fs

    # Time vector
    t = np.linspace(0, duration, N, endpoint=False)

    # Stablish which signal to inject
    signal = square_signal(t)
    # signal = model_signal(signal, N)
    # signal = random_signal(signal, N)
    # signal = pure_signal_eeg(duration, fs)

    # Apply Fourier Transform
    signal_fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, d=1/fs)
    signal_fft_magnitude = np.abs(signal_fft) / N

    # Eliminate the range of negative frequencies
    signal_fft_reduced = signal_fft[:N//2]
    frequencies_reduced = frequencies[:N//2]
    signal_fft_magnitude_reduced = signal_fft_magnitude[:N//2] 

    # Graph the entry signal
    graph_voltage_time(t, signal, title="Señal de entrada", xlabel='Tiempo [s]', ylabel='Magnitud')
    graph_voltage_frequency(frequencies_reduced, signal_fft_magnitude_reduced, title='Espectro de Frecuencias', xlabel='Frecuencia [Hz]', ylabel='Magnitud')

    # Set the band of frequencies for each signal
    alpha = (8, 13)
    beta = (13, 30)
    gamma = (30, 100)
    delta = (0.5, 4)
    theta = (4, 8)
    
    # Filter and reconstruct each signal
    alpha_t = filter_and_reconstruct(signal_fft, frequencies, alpha, N)
    beta_t = filter_and_reconstruct(signal_fft, frequencies, beta, N)
    gamma_t = filter_and_reconstruct(signal_fft, frequencies, gamma, N)
    delta_t = filter_and_reconstruct(signal_fft, frequencies, delta, N)
    theta_t = filter_and_reconstruct(signal_fft, frequencies, theta, N)

    # Plot the filtered waves in the time domain
    graph_voltage_time(t, alpha_t.real, title="Alpha en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, beta_t.real, title="Beta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, gamma_t.real, title="Gamma en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, delta_t.real, title="Delta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, theta_t.real, title="Theta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    
    # New sampling rate for interpolation
    new_fs = fs * 10
    new_t = np.linspace(0, duration, int(duration * new_fs), endpoint=True)

    # Interpolate each wave
    alpha_t_interpolated = interpolate_signal(t, alpha_t.real, new_t)
    beta_t_interpolated = interpolate_signal(t, beta_t.real, new_t)
    gamma_t_interpolated = interpolate_signal(t, gamma_t.real, new_t)
    delta_t_interpolated = interpolate_signal(t, delta_t.real, new_t)
    theta_t_interpolated = interpolate_signal(t, theta_t.real, new_t)

    # Plot the interpolated signals
    graph_voltage_time(new_t, alpha_t_interpolated, title="Alpha Interpolated", xlabel="Time [s]", ylabel="Magnitude")
    graph_voltage_time(new_t, beta_t_interpolated, title="Beta Interpolated", xlabel="Time [s]", ylabel="Magnitude")
    graph_voltage_time(new_t, gamma_t_interpolated, title="Gamma Interpolated", xlabel="Time [s]", ylabel="Magnitude")
    graph_voltage_time(new_t, delta_t_interpolated, title="Delta Interpolated", xlabel="Time [s]", ylabel="Magnitude")
    graph_voltage_time(new_t, theta_t_interpolated, title="Theta Interpolated", xlabel="Time [s]", ylabel="Magnitude")

    plt.tight_layout()
    plt.show()


# Create function with random values
def random_signal(signal, N):
    # Set fixed parameter of amplitude for random EEG(uV)
    middle_amplitude = 0 
    standard_deviation = 5

    # Save random values
    for _ in range(N):
        random_value = np.random.normal(middle_amplitude, standard_deviation)
        signal.append(random_value)


    # Asegúrate de que la longitud de la señal coincida con N
    assert len(signal) == N, "La longitud de la señal debe ser igual a N"

    return signal

def square_signal(t):
    # Set parameters of square wave
    amplitude = 3
    frequency = 10

    # Save and return square wave
    square_wave = np.array(amplitude * np.sign(np.sin(2 * np.pi * frequency * t)))
    return square_wave

def model_signal(signal, N):
    # Create a model eeg signal
    basic_eeg_signal = [
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
    ]


    eeg_signal = []

    # Repeat it until you have reached the number of samples
    repetitions = N // len(basic_eeg_signal)
    for _ in range(repetitions):
        eeg_signal.extend(basic_eeg_signal)

    # Assure that length of signal is equal to the number of samples
    assert len(signal) == N, "La longitud de la señal debe ser igual a N"

    # Convert it in an array to return it
    signal = np.array(eeg_signal)
    return signal

# Create an eeg signal with pure waves
def pure_signal_eeg(duration, fs, alpha_amp=1, alpha_frec=10, beta_amp=2, beta_frec=20,
                   gamma_amp=3, gamma_frec=40, delta_amp=4, delta_frec=2, theta_amp=5, theta_frec=5):

    # Time vector
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Set each wave as a sine wave
    senal_alfa = alpha_amp * np.sin(2 * np.pi * alpha_frec * t)
    senal_beta = beta_amp * np.sin(2 * np.pi * beta_frec * t)
    senal_gamma = gamma_amp * np.sin(2 * np.pi * gamma_frec * t)
    senal_delta = delta_amp * np.sin(2 * np.pi * delta_frec * t)
    senal_theta = theta_amp * np.sin(2 * np.pi * theta_frec * t)
    senal = senal_alfa + senal_beta + senal_gamma + senal_delta + senal_theta

    return senal

def filter_and_reconstruct(signal_fft, frequencies, band, N):
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

def graph_voltage_time(t, signal, title, xlabel, ylabel):
    # Plot given signal in th time domain
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def graph_voltage_frequency(frequencies, magnitudes, title, xlabel, ylabel='Magnitud'):
    # Plot given signal in the frquency domain
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


if __name__ == "__main__":
    main()