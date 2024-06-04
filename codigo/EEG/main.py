import numpy as np
import matplotlib.pyplot as plt

def main():
# Parámetros
    fs = 500  # Frecuencia de muestreo en Hz
    N = 1000  # Número de muestras

    signal = []
    amplitud_media = 0  # microvoltios
    desvio_estandar = 5  # microvoltios
    duration = N / fs

    # Vector de tiempo
    t = np.linspace(0, duration, N, endpoint=False)
    print(f"t: {t}")

    # signal = np.array(3 * np.sign(np.sin(2 * np.pi * 10 * t)))

    signal = random_signal(signal, N, amplitud_media, desvio_estandar)
    print(f"signal: {signal}")

    # Aplicar Transformada de Fourier
    signal_fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, d=1/fs)[:N//2]
    print(f"frequencies: {frequencies}")
    signal_fft_magnitude = np.abs(signal_fft)[:N//2] / N

    graph_voltage_time(t, signal, num=1, title="Señal de entrada", xlabel='Tiempo [s]', ylabel='Amplitud [V]')

    graph_voltage_frequency(frequencies, signal_fft_magnitude, num=2, title='Espectro de Frecuencias', xlabel='Frecuencia [Hz]', ylabel='Magnitud')

    # Filtramos las ondas específicas
    alpha = (8, 13)
    beta = (13, 30)
    gamma = (30, 100)
    delta = (0.5, 4)
    theta = (4, 8)

    # Filtramos las frecuencias correspondientes a cada onda
    alpha_indices = np.where((frequencies >= alpha[0]) & (frequencies <= alpha[1]))
    beta_indices = np.where((frequencies >= beta[0]) & (frequencies <= beta[1]))
    gamma_indices = np.where((frequencies >= gamma[0]) & (frequencies <= gamma[1]))
    theta_indices = np.where((frequencies >= delta[0]) & (frequencies <= delta[1]))
    delta_indices = np.where((frequencies >= theta[0]) & (frequencies <= theta[1]))


    # Obtener las componentes de frecuencia
    alpha_signal = signal_fft[alpha_indices]
    beta_signal = signal_fft[beta_indices]
    gamma_signal = signal_fft[gamma_indices]
    theta_signal = signal_fft[theta_indices]
    delta_signal = signal_fft[delta_indices]

    graph_voltage_frequency(frequencies[alpha_indices], np.abs(signal_fft[alpha_indices]), num=3, title="Onda Alpha", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[beta_indices], np.abs(signal_fft[beta_indices]), num=4, title="Onda Beta", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[gamma_indices], np.abs(signal_fft[gamma_indices]), num=5, title="Onda Gamma", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[theta_indices], np.abs(signal_fft[theta_indices]), num=5, title="Onda Delta", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[delta_indices], np.abs(signal_fft[delta_indices]), num=5, title="Onda Theta", xlabel="Frecuencia (Hz)")

    plt.tight_layout()
    plt.show()

def random_signal(signal, N, amplitud_media, desvio_estandar):
    for _ in range(N):
        valor_aleatorio = np.random.normal(amplitud_media, desvio_estandar)
        signal.append(valor_aleatorio)


    # Asegúrate de que la longitud de la señal coincida con N
    assert len(signal) == N, "La longitud de la señal debe ser igual a N"

    return signal


def model_signal(signal):
    # Crear una señal modelo
    senal_eeg_base = [
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


    senal_eeg = []
    numero_repeticiones = 20

    for _ in range(numero_repeticiones):
        senal_eeg.extend(senal_eeg_base)

    # Verificar la longitud de la lista
    print(len(senal_eeg))


    signal = np.array(senal_eeg)

    return signal

def graph_voltage_time(t, signal, num, title, xlabel, ylabel):
    # Graficar señal de entrada
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, num)
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def graph_voltage_frequency(frequencies, magnitudes, num, title, xlabel, ylabel='Magnitud'):
    plt.figure(figsize=(12, 6))
    plt.subplot(5, 1, num)
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


if __name__ == "__main__":
    main()