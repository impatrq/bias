import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 500  # Frecuencia de muestreo en Hz
N = 1000  # Número de muestras

signal = []
amplitud_media = 0  # microvoltios
desvio_estandar = 5  # microvoltios
duration = N / fs

'''
# Crear una señal cuadrada con amplitud de -3V a 3V
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
'''

for _ in range(N):
    valor_aleatorio = np.random.normal(amplitud_media, desvio_estandar)
    signal.append(valor_aleatorio)


# Asegúrate de que la longitud de la señal coincida con N
assert len(signal) == N, "La longitud de la señal debe ser igual a N"

# Vector de tiempo
t = np.linspace(0, duration, N, endpoint=False)
print(f"t: {t}")

# signal = np.array(3 * np.sign(np.sin(2 * np.pi * 10 * t)))

print(f"signal: {signal}")

# Aplicar Transformada de Fourier
signal_fft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, d=1/fs)[:N//2]
print(f"frequencies: {frequencies}")
signal_fft_magnitude = np.abs(signal_fft) / N

# Graficar señal de entrada
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Señal de Entrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid()

# Graficar espectro de frecuencias
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 2)
plt.plot(frequencies, signal_fft_magnitude[:N//2])
plt.title('Espectro de Frecuencias')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.grid()


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

plt.figure(figsize=(12, 6))
plt.subplot(5, 1, 2)
plt.plot(frequencies[alpha_indices], np.abs(signal_fft[alpha_indices]))
plt.title("Onda Alpha")
plt.xlabel("Frecuencia (Hz)")

plt.figure(figsize=(12, 6))
plt.subplot(5, 1, 3)
plt.plot(frequencies[beta_indices], np.abs(signal_fft[beta_indices]))
plt.title("Onda Beta")
plt.xlabel("Frecuencia (Hz)")

plt.figure(figsize=(12, 6))
plt.subplot(5, 1, 4)
plt.plot(frequencies[gamma_indices], np.abs(signal_fft[gamma_indices]))
plt.title("Onda Gamma")
plt.xlabel("Frecuencia (Hz)")

plt.figure(figsize=(12, 6))
plt.subplot(5, 1, 5)
plt.plot(frequencies[delta_indices], np.abs(signal_fft[delta_indices]))
plt.title("Onda Delta")
plt.xlabel("Frecuencia (Hz)")

plt.figure(figsize=(12, 6))
plt.subplot(5, 1, 5)
plt.plot(frequencies[theta_indices], np.abs(signal_fft[theta_indices]))
plt.title("Onda Theta")
plt.xlabel("Frecuencia (Hz)")


plt.tight_layout()
plt.show()