#Prueba de Threading 2
import threading
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import mne
from scipy.signal import butter, filtfilt, firwin, lfilter, iirfilter
from bias_reception import ReceptionBias
from bias_graphing import GraphingBias

def processing_task(signals, n, fs):
    biasProcessing = ProcessingBias(n=n, fs=fs)
    times, processed_signals = biasProcessing.process_signals(eeg_signals=signals)
    return times, processed_signals

class DSPBias:
    def __init__(self, n, fs):
        self._n = n
        self._fs = fs
        self._duration = self._n / self._fs

class ProcessingBias(DSPBias):
    def __init__(self, n, fs):
        super().__init__(n, fs)
        self._biasGraphing = GraphingBias(graph_in_terminal=True)

    def process_signals(self, eeg_signals):
        processed_signals = {}
        times = {}
        for ch, signal in eeg_signals.items():
            t, processed_signal = self.preprocess_signal(np.array(signal), ch)
            processed_signals[ch] = processed_signal
            times[ch] = t
        return times, processed_signals

    def preprocess_signal(self, eeg_signal, channel_number):
        t = np.linspace(0, self._duration, self._n, endpoint=False)
        if isinstance(eeg_signal, np.ndarray):
            signal = eeg_signal
        elif isinstance(eeg_signal, mne.epochs.Epochs):
            signal = eeg_signal.get_data(copy=True).mean(axis=0)
            t = np.linspace(0, self._duration, len(signal), endpoint=False)
        else:
            raise ValueError("Unsupported data format")

        signal_fft, frequencies, signal_fft_magnitude = self.do_fft(signal)
        signal_fft_reduced = signal_fft[:self._n//2]
        frequencies_reduced = frequencies[:self._n//2]
        signal_fft_magnitude_reduced = signal_fft_magnitude[:self._n//2]

        self._biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title=f"Input signal {channel_number}")
        self._biasGraphing.graph_signal_voltage_frequency(frequencies=frequencies_reduced, magnitudes=signal_fft_magnitude_reduced, title=f'Frequency spectrum of signal of {channel_number}')

        notch_filtered = self.apply_notch_filter(signal, freq=50.0, q=30)
        self._biasGraphing.graph_signal_voltage_time(t=t, signal=notch_filtered, title=f"Notch filtered signal {channel_number}")

        bandpass_filtered = self.apply_bandpass_filter(notch_filtered, lowcut=1, highcut=50, order=2)
        self._biasGraphing.graph_signal_voltage_time(t=t, signal=bandpass_filtered, title=f"Bandpass filtered signal {channel_number}")

        return t, bandpass_filtered

    def do_fft(self, signal):
        frequencies = np.fft.fftfreq(self._n, d=1/self._fs)
        signal_fft = np.fft.fft(signal)
        signal_fft_magnitude = np.abs(signal_fft)
        return signal_fft, frequencies, signal_fft_magnitude

    def apply_notch_filter(self, signal, freq=50.0, q=30):
        nyquist_freq = 0.5 * self._fs
        norm_freq = freq / nyquist_freq
        b, a = iirfilter(N=2, Wn=norm_freq, rs=60, btype='notch', analog=False)
        return filtfilt(b, a, signal)

    def apply_bandpass_filter(self, signal, lowcut, highcut, order=2):
        nyquist = 0.5 * self._fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

if __name__ == "__main__":
    n = 512
    fs = 256.0
    eeg_signals = {
        1: np.random.randn(n),
        2: np.random.randn(n),
        3: np.random.randn(n),
        4: np.random.randn(n)
    }
    processing_thread = threading.Thread(target=processing_task, args=(eeg_signals, n, fs))
    processing_thread.start()
