from bias_dsp import FilterBias, ProcessingBias
from bias_reception import ReceptionBias
import numpy as np
from bias_graphing import GraphingBias
from bias_motors import MotorBias
from bias_ai import AIBias
from signals import generate_synthetic_eeg, generate_synthetic_eeg_bandpower

class BiasClass:
    # Constructor
    def __init__(self, n, fs, channels, port, baudrate, timeout, save_path, saved_dataset_path, model_name, commands):
        # Define propieties for the class
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._duration = self._n / self._fs
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._save_path = save_path
        self._model_name = model_name
        self._saved_dataset_path = saved_dataset_path
        self._commands = commands 
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

    def train_ai_model(self):
        self._biasAI.collect_and_train(reception_instance=self._biasReception, filter_instance=self._biasFilter,
                                       processing_instance=self._biasProcessing, 
                                       trials_per_command=self._samples_trainig_command, save_path=self._save_path,
                                       saved_dataset_path=self._saved_dataset_path, real_data=True)

    def app_run(self):
        if self._model_name is None:
            self.train_ai_model()

        while True:
            # Receive eeg data
            #signals = self._biasReception.get_real_data(n=self._n, channels=self._number_of_channels)
            signals = generate_synthetic_eeg(n_samples=self._n, n_channels=self._number_of_channels, fs=self._fs)

            '''
            # Graph signals
            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / self._fs
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
            '''

            # Apply digital filtering
            filtered_data = self._biasFilter.filter_signals(eeg_signals=signals)

            # Calculate the time vector
            t = np.linspace(0, self._duration, self._n, endpoint=False)
            
            '''
            # Graph signals
            for ch, signal in filtered_data.items():
                # Graph filtered signal
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))
            '''

            # Process data
            times, eeg_signals = self._biasProcessing.process_signals(eeg_signals=filtered_data)

            '''
            # Plot 4 signals with its resepctive bands
            for ch, signals in eeg_signals.items():
                # Plot the interpolated signals
                for band_name, wave in signals.items():
                    self._biasGraphing.graph_signal_voltage_time(t=times[ch], signal=wave, title=f"{band_name.capitalize()} interpolated. {ch}")
            
            # Plot
            self._biasGraphing.plot_now()
            '''

            command = self._biasAI.predict_command(eeg_data=eeg_signals)
            print(f"command: {command}")

            self._biasMotor.move_if_possible(command)

