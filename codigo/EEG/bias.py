from bias_dsp import FilterBias, ProcessingBias
from bias_reception import BiasReception
import numpy as np
import graphingTerminal
import graphingPython

GRAPH_IN_TERMINAL = True

class Bias:
    def __init__(self, n, fs, channels):
        self._n = n
        self._fs = fs
        self._number_of_channels = channels
        self._duration = self._n / self._fs

    def app_run(self):
        while True:
            biasreception = BiasReception()
            signals = biasreception.get_real_data(channels=self._number_of_channels, n=self._n)

            for ch, signal in signals.items():
                t = np.arange(len(signals[ch])) / self._fs
                if GRAPH_IN_TERMINAL:
                    graphingTerminal.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))
                else:
                    graphingPython.graph_signal_voltage_time(t=t, signal=np.array(signal), title="Signal {}".format(ch))

            biasfilter = FilterBias(n=self._n, fs=self._fs, eeg_data=signals, notch=True, bandpass=True, fir=True, iir=True)

            # Apply digital filtering
            filtered_data = biasfilter.filter_signals()

            # Calculate the time vector
            t = np.linspace(0, self._duration, self._n, endpoint=False)

            for ch, signal in filtered_data.items():
                # Graph filtered signal
                if GRAPH_IN_TERMINAL:
                    graphingTerminal.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))
                else:
                    graphingPython.graph_signal_voltage_time(t=t, signal=signal, title="Filtered Signal {}".format(ch))

            biasprocessing = ProcessingBias(n=self._n, fs=self._fs, eeg_data=filtered_data)
            signals = biasprocessing.process_signals()

            if GRAPH_IN_TERMINAL:
                graphingPython.plot_now()