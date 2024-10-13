import serial
import time
import numpy as np
import json
from bias_graphing import GraphingBias

def main():
    # Set constants
    n = 1000
    fs = 500
    number_of_channels = 4
    port = '/dev/serial0'
    baudrate = 115200
    timeout = 1
    # Receive data
    biasReception = ReceptionBias(port=port, baudrate=baudrate, timeout=timeout)
    signals = biasReception.get_real_data(n=n, channels=number_of_channels)

    # Graph signals
    biasGraphing = GraphingBias(graph_in_terminal=True)
    for ch, signal in signals.items():
        t = np.arange(len(signals[ch])) / fs
        biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

class ReceptionBias:
    # Constructor
    def __init__(self, port='/dev/serial0', baudrate=115200, timeout=1):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

    # Get the data from the RP2040 Zero
    def get_real_data(self, n, channels):
        # Initialize serial communication
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
       # Loop until we have enough samples
        while len(signals['ch3']) < n: 
            if self._ser.in_waiting > 0:
                try:
                    # Read one line to detect \n character
                    data = self._ser.readline().decode('utf-8').strip()
                    eeg_data = self.process_data(data)
                    # Make an array with the data
                    if eeg_data:
                        for ch in range(channels):
                            signals[f'ch{ch}'].extend(eeg_data[f'ch{ch}'])
                except Exception as e:
                    print("Can't be decoded")


        # Check the time it takes to read
        elapsed_time = time.time() - start_time
        print(f"elapsed time: {elapsed_time}")
        # Ensure all signals have the correct length
        for ch in range(channels):
            signals[f'ch{ch}'] = signals[f'ch{ch}'][:n]

        return signals

    # Initialize serial communication
    def init_serial(self, port, baudrate, timeout):
        return serial.Serial(port, baudrate, timeout=timeout)

    # Load JSON data
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
