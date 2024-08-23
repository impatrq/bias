from bias_dsp import FilterBias, ProcessingBias
from bias_reception import ReceptionBias
import numpy as np
from bias_graphing import GraphingBias
from bias_motors import MotorBias

class Bias:
    def __init__(self, n, fs, channels, port, baudrate, timeout):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._duration = self._n / self._fs
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._biasReception = ReceptionBias(self._port, self._baudrate, self._timeout)
        self._biasFilter = FilterBias(n=self._n, fs=self._fs, notch=True, bandpass=True, fir=True, iir=True)
        self._biasProcessing = ProcessingBias(n=self._n, fs=self._fs)
        self._biasGraphing = GraphingBias(graph_in_terminal=True)
        self._biasMotor = MotorBias(echo_forward=18, trigger_forward=17, echo_backwards=23, trigger_backwards=22, echo_right=5, trigger_right=6,
                                    echo_left=27, trigger_left=24, led_forward=16, led_backwards=20, led_left=21, led_right=26, buzzer=12, moto1_in1=13, 
                                    motor1_in2=19, motor2_in1=7, motor_2_in2=8)

    def app_run(self):
        while True:
            signals = self._biasReception.get_real_data(channels=self._number_of_channels, n=self._n)

            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / self._fs
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

            # Apply digital filtering
            filtered_data = self._biasFilter.filter_signals(signals)

            # Calculate the time vector
            t = np.linspace(0, self._duration, self._n, endpoint=False)

            for ch, signal in filtered_data.items():
                # Graph filtered signal
                self._biasGraphing.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

            times, eeg_signals = self._biasProcessing.process_signals(filtered_data)

            for ch, signals in eeg_signals.items():
                # Plot the interpolated signals
                for band_name, sig in signals.items():
                    self._biasGraphing.graph_signal_voltage_time(t=times[ch], signal=sig, title=f"{band_name.capitalize()} interpolated. {ch}")
                        
            self._biasGraphing.plot_now()
            
            # command = self._biasPrediction()
            #self._biasMotor.move_if_possible(command)