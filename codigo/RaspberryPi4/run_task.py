import threading
import importlib
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def run_motor_task():
    motor_module = importlib.import_module('motor_task')
    motor_module.motor_task()

def run_processing_task():
    processing_module = importlib.import_module('processing_task')
    # Proporciona señales de ejemplo y otros parámetros necesarios
    n = 512
    fs = 256.0
    eeg_signals = {
        1: [0] * n,  # Señal de ejemplo
        2: [0] * n,  # Señal de ejemplo
        3: [0] * n,  # Señal de ejemplo
        4: [0] * n   # Señal de ejemplo
    }
    times, processed_signals = processing_module.processing_task(eeg_signals, n, fs)
    print("Processing task completed.")

def run_filter_task():
    # Parámetros de ejemplo
    fs = 256.0  # Frecuencia de muestreo
    f_notch = 50.0  # Frecuencia a eliminar con el filtro Notch
    quality_factor = 30.0  # Factor de calidad para el filtro Notch
    f_low, f_high = 1.0, 50.0  # Frecuencias para el filtro Bandpass

    # Generación de una señal de ejemplo
    t = np.linspace(0, 1, int(fs), endpoint=False)
    signal_input = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    # Filtro Notch
    def notch_filter(signal, fs, f_notch, quality_factor):
        b, a = signal.iirnotch(f_notch, quality_factor, fs)
        filtered_signal = signal.filtfilt(b, a, signal)
        return filtered_signal

    # Filtro Bandpass
    def bandpass_filter(signal, fs, f_low, f_high):
        nyq = 0.5 * fs
        low = f_low / nyq
        high = f_high / nyq
        b, a = signal.butter(5, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal)
        return filtered_signal

    # Aplicar filtro Notch
    signal_notched = notch_filter(signal_input, fs, f_notch, quality_factor)

    # Aplicar filtro Bandpass
    signal_bandpassed = bandpass_filter(signal_notched, fs, f_low, f_high)

    # Graficar resultados
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, signal_input)
    plt.title("Señal Original")
    
    plt.subplot(3, 1, 2)
    plt.plot(t, signal_notched)
    plt.title("Señal después del Filtro Notch")
    
    plt.subplot(3, 1, 3)
    plt.plot(t, signal_bandpassed)
    plt.title("Señal después del Filtro Bandpass")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Iniciar hilo para el control de motores
    motor_thread = threading.Thread(target=run_motor_task)
    motor_thread.start()

    # Iniciar hilo para el procesamiento de señales
    processing_thread = threading.Thread(target=run_processing_task)
    processing_thread.start()

    # Iniciar hilo para la tarea de filtros
    filter_thread = threading.Thread(target=run_filter_task)
    filter_thread.start()

    # Esperar a que todos los hilos terminen
    motor_thread.join()
    processing_thread.join()
    filter_thread.join()

    print("All tasks have been completed.")