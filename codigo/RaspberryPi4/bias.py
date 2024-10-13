from bias_dsp import FilterBias, ProcessingBias
from bias_reception import ReceptionBias
import numpy as np
from bias_graphing import GraphingBias
from bias_motors import MotorBias
from bias_ai import AIBias
import random
import time

class BiasClass:
    # Constructor
    def __init__(self, n, fs, channels, port, baudrate, timeout):
        # Define propieties for the class
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._duration = self._n / self._fs
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._commands = ["forward", "backwards", "left", "right"] #, "stop", "rest"]
        self._samples_trainig_command = 100

        # Create objects as propieties in order to apply the rest of the code in Bias class
        self._biasReception = ReceptionBias(self._port, self._baudrate, self._timeout)
        self._biasFilter = FilterBias(n=self._n, fs=self._fs, notch=True, bandpass=True, fir=False, iir=False)
        self._biasProcessing = ProcessingBias(n=self._n, fs=self._fs)
        self._biasGraphing = GraphingBias(graph_in_terminal=True)
        self._biasMotor = MotorBias(echo_forward=18, trigger_forward=17, echo_backwards=23, trigger_backwards=22, echo_right=5, trigger_right=6,
                                    echo_left=25, trigger_left=24, led_forward=16, led_backwards=20, led_left=21, led_right=26, buzzer=12, motor1_in1=13, 
                                    motor1_in2=19, motor2_in1=7, motor2_in2=8)
        self._biasAI = AIBias(self._n, self._fs, self._number_of_channels, self._commands)

    def train_ai_model(self, save_path, saved_dataset_path):
        self._biasAI.collect_and_train(reception_instance=self._biasReception, filter_instance=self._biasFilter,
                                       processing_instance=self._biasProcessing, 
                                       trials_per_command=self._samples_trainig_command, save_path=save_path,
                                       saved_dataset_path=saved_dataset_path, real_data=True)

    def app_run(self):
        train = input("Do you want to train model? (y/n): ")
        if train.lower() == "y":
            saved_dataset_path = None
            save_path = None
            loading_dataset = input("Do you want to load a existent dataset? (y/n): ")
            if loading_dataset.lower() == "y":
                saved_dataset_path = input("Write the name of the file where dataset was saved: ")
            else:
                save_new_dataset = input("Do you want to save the new dataset? (y/n): ")
                if save_new_dataset == "y":
                    save_path = input("Write the path where you want to save the dataset: ")
            self.train_ai_model(save_path, saved_dataset_path)

        while True:
            # Receive eeg data
            #signals = self._biasReception.get_real_data(channels=self._number_of_channels, n=self._n)
            signals = generate_synthetic_eeg(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs)

            '''
            # Graph signals
            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / self._fs
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
            '''

            # Apply digital filtering
            filtered_data = self._biasFilter.filter_signals(signals)

            # Calculate the time vector
            t = np.linspace(0, self._duration, self._n, endpoint=False)
            
            '''
            # Graph signals
            for ch, signal in filtered_data.items():
                # Graph filtered signal
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))
            '''

            # Process data
            times, eeg_signals = self._biasProcessing.process_signals(filtered_data)

            '''
            # Plot 4 signals with its resepctive bands
            for ch, signals in eeg_signals.items():
                # Plot the interpolated signals
                for band_name, sig in signals.items():
                    self._biasGraphing.graph_signal_voltage_time(t=times[ch], signal=sig, title=f"{band_name.capitalize()} interpolated. {ch}")
            
            # Plot
            self._biasGraphing.plot_now()
            '''

            command = self._biasAI.predict_command(eeg_data=eeg_signals)
            print(f"command: {command}")

            self._biasMotor.move_if_possible(command)


def generate_synthetic_eeg(n_samples, n_channels, fs):
    """
    Generate synthetic raw EEG data for multiple channels. 
    The output is a dictionary where each channel has 1000 raw samples.
    """
    t = np.linspace(0, n_samples/fs, n_samples, endpoint=False)
    data = {}

    for ch in range(n_channels):
        # Create a raw EEG signal by summing several sine waves to simulate brain activity
        signal = (
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(8, 13) * t) +  # Simulate alpha signal_wave (8-13 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(13, 30) * t) +  # Simulate beta signal_wave (13-30 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(4, 8) * t) +   # Simulate theta signal_wave (4-8 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(1, 4) * t) +   # Simulate delta signal_wave (0.5-4 Hz)
            random.randrange(0, 10) * np.sin(2 * np.pi * random.randrange(0, 50) * t)    # Simulate gamma signal_wave (30-100 Hz)
        )

        # Add random noise to simulate realistic EEG signals
        noise = np.random.normal(0, 0.5, size=t.shape)
        signal += noise

        # Store the raw signal in the dictionary
        data[ch] = signal

    return data

