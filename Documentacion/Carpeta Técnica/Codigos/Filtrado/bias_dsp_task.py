import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import mne
from scipy.signal import butter, filtfilt, firwin, lfilter, iirfilter
import threading  # Import threading
from bias_reception import ReceptionBias
from bias_graphing import GraphingBias
from bias_dsp import ProcessingBias, FilterBias
from signals import random_signal

def main():
    n = 1000
    fs = 500
    number_of_channels = 4
    duration = n / fs

    # Initialize objects
    biasReception = ReceptionBias()
    biasGraphing = GraphingBias(graph_in_terminal=False)
    biasFilter = FilterBias(n=n, fs=fs, notch=True, bandpass=True, fir=True, iir=True)
    biasProcessing = ProcessingBias(n=n, fs=fs)

    # Thread for receiving data
    reception_thread = threading.Thread(target=receive_data, args=(biasReception, number_of_channels, n, biasGraphing, fs))

    # Thread for filtering and processing data
    processing_thread = threading.Thread(target=process_data, args=(biasReception, n, fs, biasFilter, biasProcessing, biasGraphing))

    # Start both threads
    reception_thread.start()
    processing_thread.start()

    # Wait for both threads to finish
    reception_thread.join()
    processing_thread.join()

# Function to receive data
def receive_data(biasReception, number_of_channels, n, biasGraphing, fs):
    # Receive data from RP2040 Zero
    signals = biasReception.get_real_data(channels=number_of_channels, n=n)
    # Graph the signals
    for ch, signal in signals.items():
        t = np.arange(len(signals[ch])) / fs
        biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

# Function to process data (filter and further processing)
def process_data(biasReception, n, fs, biasFilter, biasProcessing, biasGraphing):
    # Receive data from RP2040 Zero
    signals = biasReception.get_real_data(channels=4, n=n)

    # Apply digital filtering
    filtered_data = biasFilter.filter_signals(signals)

    # Calculate the time vector
    duration = n / fs
    t = np.linspace(0, duration, n, endpoint=False)

    # Graph filtered signals
    for ch, signal in filtered_data.items():
        biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

    # Process filtered signals
    times, processed_signals = biasProcessing.process_signals(filtered_data)

    # Graph processed signals
    for ch, signal in processed_signals.items():
        biasGraphing.graph_signal_voltage_time(t=times[ch], signal=signal, title="Processed Signal {}".format(ch))

    # Plot final graphs
    biasGraphing.plot_now()

# Classes remain unchanged
# (DSPBias, ProcessingBias, FilterBias, etc.)

if __name__ == "__main__":
    main()
