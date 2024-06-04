import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 500  # Frecuencia de muestreo en Hz
N = 1000  # Número de muestras

signal = []
amplitud_media = 0  # microvoltios
desvio_estandar = 5  # microvoltios

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
'''

print(signal)

# Asegúrate de que la longitud de la señal coincida con N
assert len(signal) == N, "La longitud de la señal debe ser igual a N"

# Vector de tiempo
t = np.linspace(0, N / fs, N, endpoint=False)

# Aplicar Transformada de Fourier
signal_fft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, 1/fs)
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
plt.subplot(2, 1, 2)
plt.plot(frequencies[:N//2], signal_fft_magnitude[:N//2])
plt.title('Espectro de Frecuencias')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.grid()

plt.tight_layout()
plt.show()