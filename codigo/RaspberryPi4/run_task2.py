import threading
import importlib
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class MotorTask:
    def __init__(self):
        self.motor_module = importlib.import_module('motor_task')

    def run(self):
        self.motor_module.motor_task()

class ProcessingTask:
    def __init__(self, n=512, fs=256.0):
        self.processing_module = importlib.import_module('processing_task')
        self.n = n
        self.fs = fs
        self.eeg_signals = {
            1: [0] * self.n,
            2: [0] * self.n,
            3: [0] * self.n,
            4: [0] * self.n
        }

    def run(self):
        times, processed_signals = self.processing_module.processing_task(self.eeg_signals, self.n, self.fs)
        print("Processing task completed.")

class FilterTask:
    def __init__(self, fs=256.0, f_notch=50.0, quality_factor=30.0, f_low=1.0, f_high=50.0):
        self.fs = fs
        self.f_notch = f_notch
        self.quality_factor = quality_factor
        self.f_low = f_low
        self.f_high = f_high

    def generate_signal(self):
        t = np.linspace(0, 1, int(self.fs), endpoint=False)
        return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    def notch_filter(self, signal):
        b, a = signal.iirnotch(self.f_notch, self.quality_factor, self.fs)
        return signal.filtfilt(b, a, signal)

    def bandpass_filter(self, signal):
        nyq = 0.5 * self.fs
        low = self.f_low / nyq
        high = self.f_high / nyq
        b, a = signal.butter(5, [low, high], btype='band')
        return signal.filtfilt(b, a, signal)

    def plot_results(self, t, signal_input, signal_notched, signal_bandpassed):
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

    def run(self):
        t = np.linspace(0, 1, int(self.fs), endpoint=False)
        signal_input = self.generate_signal()
        signal_notched = self.notch_filter(signal_input)
        signal_bandpassed = self.bandpass_filter(signal_notched)
        self.plot_results(t, signal_input, signal_notched, signal_bandpassed)

class EEGProcessor:
    def __init__(self):
        self.motor_task = MotorTask()
        self.processing_task = ProcessingTask()
        self.filter_task = FilterTask()

    def run_all_tasks(self):
        motor_thread = threading.Thread(target=self.motor_task.run)
        processing_thread = threading.Thread(target=self.processing_task.run)
        filter_thread = threading.Thread(target=self.filter_task.run)

        motor_thread.start()
        processing_thread.start()
        filter_thread.start()

        motor_thread.join()
        processing_thread.join()
        filter_thread.join()

        print("All tasks have been completed.")

if __name__ == "__main__":
    processor = EEGProcessor()
    processor.run_all_tasks()