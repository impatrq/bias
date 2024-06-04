import numpy as np
import matplotlib.pyplot as plt
import scipy

def main():
# Parámetros
    fs = 500
    ts = 1/fs
      # Frecuencia de muestreo en Hz
    N = 5000  # Número de muestras

    signal = []
    amplitud_media = 0  # microvoltios
    desvio_estandar = 5  # microvoltios
    duration = N / fs

    # Vector de tiempo
    t = np.linspace(0, duration, N, endpoint=False)
    # print(f"t: {t}")

    # signal = np.array(3 * np.sign(np.sin(2 * np.pi * 10 * t)))
    # signal = model_signal(signal, N)
    # signal = random_signal(signal, N, amplitud_media, desvio_estandar)
    signal = pure_signal_eeg(duration, fs)

    #print(f"signal: {signal}")

    # Aplicar Transformada de Fourier
    signal_fft = np.fft.fft(signal)
    #print(f"signal_fft: {signal_fft}")
    frequencies = np.fft.fftfreq(N, d=1/fs)[:N//2]
    #print(f"frequencies: {frequencies}")
    signal_fft_magnitude = np.abs(signal_fft)[:N//2] / N

    graph_voltage_time(t, signal, num=1, title="Señal de entrada", xlabel='Tiempo [s]', ylabel='Magnitud')

    graph_voltage_frequency(frequencies, signal_fft_magnitude, num=1, title='Espectro de Frecuencias', xlabel='Frecuencia [Hz]', ylabel='Magnitud')

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

    alpha_t = np.fft.ifft(alpha_signal + alpha_signal[::-1])
    t_alpha = np.linspace(0, duration, len(alpha_t))
    graph_voltage_time(t_alpha, alpha_t.real, num=1, title="Alpha en función del tiempo", xlabel="Tiempo (s)", ylabel="Magnitud")

    beta_t = np.fft.ifft(beta_signal + beta_signal[::-1])
    t_beta = np.linspace(0, duration, len(beta_t))
    graph_voltage_time(t_beta, beta_t.real, num=1, title="Beta en función del tiempo", xlabel="Tiempo (s)", ylabel="Magnitud")

    gamma_t = np.fft.ifft(gamma_signal + gamma_signal[::-1])
    t_gamma = np.linspace(0, duration, len(gamma_t))
    graph_voltage_time(t_gamma, gamma_t.real, num=1, title="Gamma en función del tiempo", xlabel="Tiempo (s)", ylabel="Magnitud")

    delta_t = np.fft.ifft(delta_signal + delta_signal[::-1])
    t_delta = np.linspace(0, duration, len(delta_t))
    graph_voltage_time(t_delta, delta_t.real, num=1, title="Delta en función del tiempo", xlabel="Tiempo (s)", ylabel="Magnitud")

    theta_t = np.fft.ifft(theta_signal + theta_signal[::-1])
    t_theta = np.linspace(0, duration, len(theta_t))
    graph_voltage_time(t_theta, theta_t.real, num=1, title="Theta en función del tiempo", xlabel="Tiempo (s)", ylabel="Magnitud")

    #graph_voltage_frequency(frequencies[alpha_indices], np.abs(signal_fft[alpha_indices]), num=3, title="Onda Alpha", xlabel="Frecuencia (Hz)")
    #graph_voltage_frequency(frequencies[beta_indices], np.abs(signal_fft[beta_indices]), num=4, title="Onda Beta", xlabel="Frecuencia (Hz)")
    #graph_voltage_frequency(frequencies[gamma_indices], np.abs(signal_fft[gamma_indices]), num=5, title="Onda Gamma", xlabel="Frecuencia (Hz)")
    #graph_voltage_frequency(frequencies[theta_indices], np.abs(signal_fft[theta_indices]), num=5, title="Onda Delta", xlabel="Frecuencia (Hz)")
    #graph_voltage_frequency(frequencies[delta_indices], np.abs(signal_fft[delta_indices]), num=5, title="Onda Theta", xlabel="Frecuencia (Hz)")

    plt.tight_layout()
    plt.show()

def random_signal(signal, N, amplitud_media, desvio_estandar):
    for _ in range(N):
        valor_aleatorio = np.random.normal(amplitud_media, desvio_estandar)
        signal.append(valor_aleatorio)


    # Asegúrate de que la longitud de la señal coincida con N
    assert len(signal) == N, "La longitud de la señal debe ser igual a N"

    return signal

def model_signal(signal, N):
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
    numero_repeticiones = N // len(senal_eeg_base)

    for _ in range(numero_repeticiones):
        senal_eeg.extend(senal_eeg_base)

    # Verificar la longitud de la lista
    print(len(senal_eeg))


    signal = np.array(senal_eeg)

    return signal

def pure_signal_eeg(duracion, fs, alpha_amp=1, alpha_frec=10, beta_amp=2, beta_frec=20,
                   gamma_amp=3, gamma_frec=40, delta_amp=4, delta_frec=2, theta_amp=5, theta_frec=5):
  # Generación de tiempo
  t = np.linspace(0, duracion, int(duracion * fs))

  # Generación de componentes sinusoidales
  senal_alfa = alpha_amp * np.sin(2 * np.pi * alpha_frec * t)
  senal_beta = beta_amp * np.sin(2 * np.pi * beta_frec * t)
  senal_gamma = gamma_amp * np.sin(2 * np.pi * gamma_frec * t)
  senal_delta = delta_amp * np.sin(2 * np.pi * delta_frec * t)
  senal_theta = theta_amp * np.sin(2 * np.pi * theta_frec * t)

  # Suma de componentes
  senal = senal_alfa + senal_beta + senal_gamma + senal_delta + senal_theta

  return senal

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
    plt.subplot(2, 1, num)
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


if __name__ == "__main__":
    main()