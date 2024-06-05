import numpy as np
import matplotlib.pyplot as plt
import scipy

def main():
# Parámetros
    fs = 500
    #ts = 1/fs
    # Frecuencia de muestreo en Hz
    N = 10000  # Número de muestras

    # signal = []
    #amplitud_media = 0  # microvoltios
    #desvio_estandar = 5  # microvoltios
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

    graph_voltage_time(t, signal, title="Señal de entrada", xlabel='Tiempo [s]', ylabel='Magnitud')
    graph_voltage_frequency(frequencies, signal_fft_magnitude, title='Espectro de Frecuencias', xlabel='Frecuencia [Hz]', ylabel='Magnitud')

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
    delta_indices = np.where((frequencies >= delta[0]) & (frequencies <= delta[1]))
    theta_indices = np.where((frequencies >= theta[0]) & (frequencies <= theta[1]))

    # Inicializar señales filtradas
    alpha_signal = np.zeros_like(signal_fft)
    beta_signal = np.zeros_like(signal_fft)
    gamma_signal = np.zeros_like(signal_fft)
    delta_signal = np.zeros_like(signal_fft)
    theta_signal = np.zeros_like(signal_fft)

    '''
    # Obtener las componentes de frecuencia
    alpha_signal = signal_fft[alpha_indices]
    beta_signal = signal_fft[beta_indices]
    gamma_signal = signal_fft[gamma_indices]
    delta_signal = signal_fft[delta_indices]
    theta_signal = signal_fft[theta_indices]
    '''
    # Asignar componentes de frecuencia filtradas
    alpha_signal[alpha_indices] = signal_fft[alpha_indices]
    beta_signal[beta_indices] = signal_fft[beta_indices]
    gamma_signal[gamma_indices] = signal_fft[gamma_indices]
    delta_signal[delta_indices] = signal_fft[delta_indices]
    theta_signal[theta_indices] = signal_fft[theta_indices]
    
    # Reconstruir las señales en el dominio del tiempo utilizando iFFT
    alpha_t = np.fft.ifft(alpha_signal)
    beta_t = np.fft.ifft(beta_signal)
    gamma_t = np.fft.ifft(gamma_signal)
    delta_t = np.fft.ifft(delta_signal)
    theta_t = np.fft.ifft(theta_signal)

    # Graficar las señales filtradas en el dominio del tiempo
    graph_voltage_time(t, alpha_t.real, title="Alpha en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, beta_t.real, title="Beta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, gamma_t.real, title="Gamma en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, delta_t.real, title="Delta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    graph_voltage_time(t, theta_t.real, title="Theta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")

    '''
    alpha_t = np.fft.ifft(alpha_signal)
    t_alpha = np.linspace(0, duration, len(alpha_t))
    graph_voltage_time(t_alpha, alpha_t.real, title="Alpha en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")

    beta_t = np.fft.ifft(beta_signal + beta_signal[::-1])
    t_beta = np.linspace(0, duration, len(beta_t))
    graph_voltage_time(t_beta, beta_t.real, title="Beta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")

    gamma_t = np.fft.ifft(gamma_signal + gamma_signal[::-1])
    t_gamma = np.linspace(0, duration, len(gamma_t))
    graph_voltage_time(t_gamma, gamma_t.real, title="Gamma en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")

    delta_t = np.fft.ifft(delta_signal + delta_signal[::-1])
    t_delta = np.linspace(0, duration, len(delta_t))
    graph_voltage_time(t_delta, delta_t.real, title="Delta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")

    theta_t = np.fft.ifft(theta_signal + theta_signal[::-1])
    t_theta = np.linspace(0, duration, len(theta_t))
    graph_voltage_time(t_theta, theta_t.real, title="Theta en función del tiempo", xlabel="Tiempo [s]", ylabel="Magnitud")
    '''
    graph_voltage_frequency(frequencies[alpha_indices], np.abs(signal_fft[alpha_indices]), title="Onda Alpha", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[beta_indices], np.abs(signal_fft[beta_indices]), title="Onda Beta", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[gamma_indices], np.abs(signal_fft[gamma_indices]), title="Onda Gamma", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[delta_indices], np.abs(signal_fft[delta_indices]), title="Onda Delta", xlabel="Frecuencia (Hz)")
    graph_voltage_frequency(frequencies[theta_indices], np.abs(signal_fft[theta_indices]), title="Onda Theta", xlabel="Frecuencia (Hz)")
    
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
    t = np.linspace(0, duracion, int(duracion * fs))
    senal_alfa = alpha_amp * np.sin(2 * np.pi * alpha_frec * t)
    senal_beta = beta_amp * np.sin(2 * np.pi * beta_frec * t)
    senal_gamma = gamma_amp * np.sin(2 * np.pi * gamma_frec * t)
    senal_delta = delta_amp * np.sin(2 * np.pi * delta_frec * t)
    senal_theta = theta_amp * np.sin(2 * np.pi * theta_frec * t)
    senal = senal_alfa + senal_beta + senal_gamma + senal_delta + senal_theta
    return senal

def graph_voltage_time(t, signal, title, xlabel, ylabel):
    # Graficar señal de entrada
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def graph_voltage_frequency(frequencies, magnitudes, title, xlabel, ylabel='Magnitud'):
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, magnitudes)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


if __name__ == "__main__":
    main()