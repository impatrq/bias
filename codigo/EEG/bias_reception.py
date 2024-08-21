import serial
import time
import numpy as np
import graphingPython  # type: ignore
import graphingTerminal
import json
import filtering

GRAPH_IN_TERMINAL = True

def main():
    # Set constants
    n = 1000
    fs = 500
    number_of_channels = 4
    biasreception = BiasReception()
    signals = biasreception.get_real_data(channels=number_of_channels, n=n)
    for ch, signal in signals.items():
        t = np.arange(len(signals[ch])) / fs
        if GRAPH_IN_TERMINAL:
            graphingTerminal.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
        else:
            graphingPython.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

class BiasReception:
    def __init__(self, port='/dev/serial0', baudrate=115200, timeout=1):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        
    def get_real_combined_data(self, channels, n, fs, filter):
        # Get real data of signals
        real_eeg_signals = self.get_real_data(channels=channels, n=n)
        
        # Filter if necessary
        if filter:
            filtered_signals = filtering.filter_signals(eeg_signals=real_eeg_signals, fs=fs)
            combined_signal = self.combine_signals(filtered_signals)
        else:
            combined_signal = self.combine_signals(real_eeg_signals)

        return combined_signal

    # Get the data from the RP2040 Zero
    def get_real_data(self, channels, n):
        self._ser = self.init_serial(self._port, self._baudrate, self._timeout)
        try:
            # Capture the signal from the UART
            real_eeg_signals = self.capture_signals(channels=channels, n=n)
            return real_eeg_signals
        finally:
            self._ser.close()

    def capture_signals(self, channels, n):
        # Initialize variables
        signals = {f'ch{ch}': [] for ch in range(channels)}
        start_time = time.time()

        while len(signals['ch3']) < n:  # Loop until we have enough samples
            if self._ser.in_waiting > 0:
                try:
                    data = self._ser.readline().decode('utf-8').strip()
                    eeg_data = self.process_data(data)
                    if eeg_data:
                        for ch in range(channels):
                            signals[f'ch{ch}'].extend(eeg_data[f'ch{ch}'])
                except Exception as e:
                    print("Can't be decoded")


        # Check the time it takes to read
        elapsed_time = time.time() - start_time
        print(f"elapsed time: {elapsed_time}")    # Ensure all signals have the correct length
        for ch in range(channels):
            signals[f'ch{ch}'] = signals[f'ch{ch}'][:n]

        return signals

    def combine_signals(self, signals):
        combined_signal = np.mean([signals[f'ch{ch}'] for ch in range(len(signals))], axis=0)
        return combined_signal

    def init_serial(self, port, baudrate, timeout):
        return serial.Serial(port, baudrate, timeout=timeout)

    def process_data(self, data):
        try:
            json_data = json.loads(data)
            print(json_data)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None

if __name__ == "__main__":
    main()