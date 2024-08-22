from bias_dsp import FilterBias, ProcessingBias
from bias_reception import BiasReception
import numpy as np
import graphingTerminal
import graphingPython

GRAPH_IN_TERMINAL = True

class Bias:
    def __init__(self, n, fs, channels, port, baudrate, timeout):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._duration = self._n / self._fs
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

    def app_run(self):
        biasreception = BiasReception(self._port, self._baudrate, self._timeout)
        biasfilter = FilterBias(n=self._n, fs=self._fs, notch=True, bandpass=True, fir=True, iir=True)
        biasprocessing = ProcessingBias(n=self._n, fs=self._fs)

        while True:
            signals = biasreception.get_real_data(channels=self._number_of_channels, n=self._n)

            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / self._fs
                if GRAPH_IN_TERMINAL:
                    graphingTerminal.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
                else:
                    graphingPython.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

            # Apply digital filtering
            filtered_data = biasfilter.filter_signals(signals)

            # Calculate the time vector
            t = np.linspace(0, self._duration, self._n, endpoint=False)

            for ch, signal in filtered_data.items():
                # Graph filtered signal
                if GRAPH_IN_TERMINAL:
                    graphingTerminal.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))
                else:
                    graphingPython.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

            times, eeg_signals = biasprocessing.process_signals(filtered_data)

            for ch, signals in eeg_signals.items():
                # Plot the interpolated signals
                for band_name, sig in signals.items():
                    if GRAPH_IN_TERMINAL:
                        graphingTerminal.graph_signal_voltage_time(t=times[ch], signal=sig, title=f"{band_name.capitalize()} interpolated. {ch}")
                    else:
                        graphingPython.graph_signal_voltage_time(t=times[ch], signal=sig, title=f"{band_name.capitalize()} interpolated. {ch}")

                    if not GRAPH_IN_TERMINAL:
                        graphingPython.plot_now()