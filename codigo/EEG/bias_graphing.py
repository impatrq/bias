import matplotlib.pyplot as plt
import plotext

class GraphingBias:
    # Constructor
    def __init__(self, graph_in_terminal):
        self._graph_in_terminal = graph_in_terminal

    # Graph signal in function of time
    def graph_signal_voltage_time(self, t, signal, title):
        if not self._graph_in_terminal:
            # Plot given signal in th time domain
            plt.figure(figsize=(12, 6))
            if signal.ndim == 1:
                plt.plot(t, signal)
            else:
                for i in range(signal.shape[0]):
                    plt.plot(t, signal[i])
            # Set graph paramters
            plt.title(title)
            plt.xlabel("Time [s]")
            plt.ylabel("Magnitude")
            plt.grid()

        else:
            plotext.clear_data()
            plotext.plot(t, signal)
            # Set graph parameters
            plotext.title(title)
            plotext.xlabel("Time [s]")
            plotext.ylabel("Magnitude")
            plotext.show()

    def graph_signal_voltage_frequency(self, frequencies, magnitudes, title):
        if not self._graph_in_terminal:
            # Plot given signal in the frquency domain
            plt.figure(figsize=(12, 6))
            plt.plot(frequencies, magnitudes)
            # Set graph parameters
            plt.title(title)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.grid()

        else:
            plotext.clear_data()
            plotext.plot(frequencies, magnitudes)
            # Set graph parameters
            plotext.title(title)
            plotext.xlabel("Frequency [Hz]")
            plotext.ylabel("Magnitude")
            plotext.show()

    def plot_now(self):
        if not self._graph_in_terminal:
            plt.tight_layout()
            plt.show()
        else:
            pass